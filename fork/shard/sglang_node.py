"""SglangNodeRuntime — the per-node block runtime (Cairn's rung-2/3 implementation).

⚠️ GPU-UNVERIFIED HEAD START. The structure, the architecture decision, and the
concretely-writable parts (heartbeat, KV bookkeeping, config) are here; the SGLang
block-forward integration is marked TODO and MUST be iterated on a real GPU with a
pinned SGLang version. Do not assume it runs as-is.

Architecture decision (load-bearing — record it):

  SGLang HAS native pipeline parallelism (`--pp-size`/`--nnodes`, NCCL between stages).
  **Cairn does NOT use it across nodes.** SGLang's PP is one coordinated launch whose
  stages hand off activations over NCCL collectives — which *hang or abort when any rank
  vanishes*. Our nodes are independent spot instances that get reclaimed mid-decode; the
  whole product is reassign-one-block + replay-KV + keep-serving (blast radius 1/N). A
  NCCL-coupled pipeline has blast radius = total. So each node runs SGLang loaded with
  ONLY its contiguous block of layers and exposes `forward(hidden, kv_meta) → hidden`;
  **Cairn's own wire/transport/scheduler/recovery stitch the stages** (spec §0 inv #1,
  §5; handoff §3). We keep SGLang's kernels/attention/paging/quant — not its transport.

Integration point (the rung-2 work): load a *layer-range subset* of the model and drive
a single block's forward with per-seq paged KV. Candidate paths, to settle on a GPU:
  (A) Drive SGLang's per-stage model runner directly (it already loads a layer subset
      for its own PP) but bypass its NCCL send/recv — feed our `hidden` in, take the
      stage output out. Deepest, most reuse.
  (B) A thin custom runner that loads the layer slice + SGLang's RadixAttention/paged-KV
      for this block's cache. More code, fewer SGLang-internal assumptions.
  (C) Rung-2 stepping stone: a transformers reference block forward (no SGLang) to prove
      split-correctness + KV-replay cheaply, then swap to (A)/(B) for performance.
"""

from __future__ import annotations

from typing import Any, Dict

from .node import LayerRange, NodeRuntime

try:  # keep this module importable on CPU (for structural checks) — deps load lazily
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class SglangNotAvailable(RuntimeError):
    pass


class SglangNodeRuntime(NodeRuntime):
    """Serve one contiguous block of layers via SGLang (rung 2/3, GPU).

    PROVEN on a real L4 (2026-06-21): split == unsplit token-for-token (inv #1) over multi-step
    greedy decode, on SGLang's own flashinfer kernels + paged KV — by driving SGLang's per-layer
    forward directly (path A), with its NCCL pipeline-parallel transport bypassed (Cairn's wire
    stitches the stages instead; spec §0 inv #1, handoff §3).

    The seam:
      - `load_shard` builds an in-process SGLang `ModelRunner` (tp=1, pp=1 — no NCCL) that loads the
        model + a paged-KV pool + the flashinfer attention backend, and keeps a view of this block's
        layers `[start, end)`. SGLang's distributed state is GLOBAL per process, so all runtimes in
        ONE process SHARE one ModelRunner (`_SHARED`): a real node is its own process → its own
        runner; the single-GPU split test = N runtimes sharing one. The KV pool is layer-partitioned,
        so blocks never collide.
      - `forward` runs `layers[start:end]` over the incoming hidden (embedding the ids on stage 0),
        keeping this block's paged KV per seq via an SGLang `ScheduleBatch` (first call = prefill /
        extend, later calls = decode). SGLang uses the residual-stream pattern (each layer returns
        `(hidden, residual)`); at a non-tail boundary we FOLD them — `hidden + residual` — so the wire
        carries ONE tensor and the next stage resumes with `residual=None` (algebraically exact). The
        tail applies `norm` + `lm_head` → next-token logits.
    """

    _SHARED: Dict = {}   # (model, device) -> ModelRunner; one per process (SGLang global dist state)

    def __init__(self, model: str, layer_range: LayerRange, device: str = "cuda:0",
                 quant=None) -> None:
        super().__init__(model, layer_range, device)
        self.quant = quant                       # None = the model's native dtype (don't force a quant)
        parts = device.split(":")                # tolerate "cpu"/"cuda"/"cuda:N" without crashing (L1)
        self.device_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        self._runner = None
        self._inner = None                       # the inner decoder model (.layers/.embed_tokens/.norm)
        self._batches: Dict[str, Any] = {}       # seq_id -> SGLang ScheduleBatch (KV lifecycle)
        self._steps: Dict[str, int] = {}         # seq_id -> forward count (0 = needs prefill)

    # ---- load: build/attach the in-process ModelRunner + this block's layer view ----
    def load_shard(self) -> None:
        if not _HAS_TORCH:
            raise SglangNotAvailable(
                "torch+sglang required; install infra/requirements-gpu.txt on a GPU box"
            )
        import os
        import socket
        from sglang.srt.server_args import ServerArgs
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner

        key = (self.model, self.device)
        runner = SglangNodeRuntime._SHARED.get(key)
        if runner is None:
            mem = float(os.environ.get("CAIRN_SGLANG_MEM_FRACTION", "0.8"))
            sock = socket.socket(); sock.bind(("", 0)); port = sock.getsockname()[1]; sock.close()
            sa = ServerArgs(model_path=self.model, tp_size=1, pp_size=1, mem_fraction_static=mem,
                            disable_cuda_graph=True, trust_remote_code=True)
            runner = ModelRunner(
                model_config=ModelConfig.from_server_args(sa), mem_fraction_static=mem,
                gpu_id=self.device_index, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                nccl_port=port, server_args=sa,
            )
            SglangNodeRuntime._SHARED[key] = runner
        self._runner = runner
        self._inner = runner.model.model
        n = len(self._inner.layers)
        s, e = self.layer_range.start, self.layer_range.end   # end EXCLUSIVE (LayerRange convention)
        if not (0 <= s < e <= n):
            raise ValueError(f"layer_range [{s},{e}) out of [0,{n}] for {self.model}")
        self._start, self._end, self._n = s, e, n
        self._is_embed = (s == 0)                # stage 0 embeds the token ids
        self._is_tail = (e == n)                 # last stage applies norm + lm_head

    # ---- forward: run this block's layers over `hidden_states`; per-seq paged KV stays here ----
    def forward(self, hidden_states: Any, kv_meta: Dict[str, Any]) -> Any:
        if self._runner is None:
            raise RuntimeError("SglangNodeRuntime.forward: call load_shard() first")
        import torch
        from sglang.srt.managers.schedule_batch import ScheduleBatch, Req
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        seq = kv_meta["seq"]
        rid = f"{seq}::{self._start}-{self._end}"        # unique per (stream, block) on the shared pool
        x = hidden_states                                # [1,S] token ids (embed) or [1,S,H] hidden
        s_len = x.shape[1]
        step = self._steps.get(seq, 0)
        if step == 0:                                    # prefill (extend): ingest the prompt / replay
            ids = x[0].tolist() if self._is_embed else [0] * s_len   # mid blocks don't embed → dummy ids
            req = Req(rid=rid, origin_input_text="", origin_input_ids=ids,
                      sampling_params=SamplingParams(temperature=0.0, max_new_tokens=1))
            req.prefix_indices = []
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(ids)
            req.logprob_start_len = len(ids) - 1
            batch = ScheduleBatch.init_new(
                reqs=[req], req_to_token_pool=self._runner.req_to_token_pool,
                token_to_kv_pool_allocator=self._runner.token_to_kv_pool_allocator, tree_cache=None,
                model_config=self._runner.model_config, enable_overlap=False,
                spec_algorithm=SpeculativeAlgorithm.NONE, enable_custom_logit_processor=False,
            )
            batch.prepare_for_extend()
            self._batches[seq] = batch
        else:                                            # decode: one token
            batch = self._batches[seq]
            tok = int(x[0, -1]) if self._is_embed else 0
            batch.output_ids = torch.tensor([tok], dtype=torch.int64, device=x.device)
            batch.prepare_for_decode()
        fb = ForwardBatch.init_new(batch.get_model_worker_batch(), self._runner)
        self._runner.attn_backend.init_forward_metadata(fb)   # flashinfer metadata for this batch

        if self._is_embed:
            hidden = self._inner.embed_tokens(fb.input_ids)
            residual = None
        else:
            hidden = x.reshape(-1, x.shape[-1])          # [1,S,H] -> flat [S,H] (SGLang token layout)
            residual = None
        for i in range(self._start, self._end):
            hidden, residual = self._inner.layers[i](fb.positions, hidden, fb, residual)
        self._steps[seq] = step + 1

        if self._is_tail:
            hidden, _ = self._inner.norm(hidden, residual)
            logits = self._runner.model.logits_processor(
                fb.input_ids, hidden, self._runner.model.lm_head, fb
            ).next_token_logits
            return logits.reshape(1, -1, logits.shape[-1])   # [1, 1, V] — the harness takes [:, -1]
        folded = hidden + residual if residual is not None else hidden
        return folded.reshape(1, s_len, -1)                  # [1, S, H] hand-off to the next stage

    def free_seq(self, seq_id: str) -> None:
        """Drop a seq's KV on this block (on completion, or before a replay-rebuild)."""
        self._steps.pop(seq_id, None)
        batch = self._batches.pop(seq_id, None)
        if batch is not None:
            try:                                          # best-effort release back to the pools
                self._runner.req_to_token_pool.free(batch.req_pool_indices)
                self._runner.token_to_kv_pool_allocator.free(batch.out_cache_loc)
            except Exception:
                pass

    def kv_tokens(self, seq_id: str) -> int:
        batch = self._batches.get(seq_id)
        try:
            return int(batch.seq_lens[0]) if batch is not None else 0
        except Exception:
            return self._steps.get(seq_id, 0)

    # ---- heartbeat: real VRAM/liveness (concretely writable now) ----
    def heartbeat(self) -> dict:
        info: dict = {
            "loaded": self._runner is not None,
            "layers": [self.layer_range.start, self.layer_range.end],
            "device": self.device,
            "kv_seqs": len(self._kv_seqs),
        }
        if _HAS_TORCH and torch.cuda.is_available():
            try:
                free, total = torch.cuda.mem_get_info(self.device_index)
                info["vram_used_gb"] = round((total - free) / 2**30, 2)
                info["vram_total_gb"] = round(total / 2**30, 2)
                info["alive"] = True
            except Exception as e:  # bad ordinal / driver error → unhealthy, never throw (L2)
                info["alive"] = False
                info["note"] = f"VRAM probe failed: {e}"
        else:
            info["alive"] = False
            info["note"] = "no CUDA — this runtime is GPU-only (rung 2/3)"
        return info
