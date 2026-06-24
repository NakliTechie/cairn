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

from typing import Any, Dict, Optional

from .node import LayerRange, NodeRuntime

try:  # keep this module importable on CPU (for structural checks) — deps load lazily
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class SglangNotAvailable(RuntimeError):
    pass


class _CairnFakePPGroup:
    """Stand-in for sglang's pipeline-parallel GroupCoordinator (path-β loader).

    sglang's per-model `make_layers(..., pp_rank, pp_size)` reads from `get_pp_group()` and
    uses `rank_in_group`/`world_size` to compute the layer slice for THIS rank. Layers outside
    the slice become `PPMissingLayer` placeholders (no weights resident in VRAM).

    Cairn's invariant (2026-06-19): no NCCL between nodes. We get the loader-side partition by
    monkey-patching `parallel_state.get_pp_group` to return THIS fake. sglang's distributed init
    stays world_size=1 (no inter-node TCP rendezvous, no NCCL groups). Forward goes through
    Cairn's own wire (per-layer driven), so the unused collective methods below are never called.
    Stubbed regardless so a mistaken call crashes loudly rather than hanging on a real NCCL op.
    """

    def __init__(self, rank: int, size: int) -> None:
        self.rank_in_group = rank
        self.world_size = size
        self.ranks = list(range(size))
        self.device_group = None
        self.cpu_group = None

    @property
    def is_first_rank(self) -> bool:
        return self.rank_in_group == 0

    @property
    def is_last_rank(self) -> bool:
        return self.rank_in_group == self.world_size - 1

    @property
    def next_rank(self) -> int:
        return self.ranks[(self.rank_in_group + 1) % self.world_size]

    @property
    def prev_rank(self) -> int:
        return self.ranks[(self.rank_in_group - 1) % self.world_size]

    def _refuse(self, name: str):
        raise RuntimeError(
            f"_CairnFakePPGroup.{name}() called: Cairn drives layers directly, NCCL inter-stage "
            "collectives must never fire (would hang on a vanished spot node, spec inv §0)."
        )

    def all_reduce(self, *a, **kw):  return self._refuse("all_reduce")
    def all_gather(self, *a, **kw):  return self._refuse("all_gather")
    def broadcast(self, *a, **kw):   return self._refuse("broadcast")
    def send(self, *a, **kw):        return self._refuse("send")
    def recv(self, *a, **kw):        return self._refuse("recv")
    def barrier(self, *a, **kw):     pass     # tolerate barrier no-op; some init paths call it


def _set_pp_identity(group: Any, rank: int, size: int) -> None:
    """Override an sglang pp GroupCoordinator's *partition identity* in place (path-β core).

    sglang decides each box's layer slice from `get_pp_group().rank_in_group` / `.world_size`:
    `DeepseekV*Model.__init__` reads them and passes `pp_rank`/`pp_size` to `make_layers`, which
    calls `get_pp_indices` to compute `[start,end)`. We must make those read Cairn's `(k, N)`.

    Why mutate-in-place from a `load_model` hook (not replace `get_pp_group` before ModelRunner):
    `ModelRunner.__init__` runs `init_torch_distributed()` → `initialize_model_parallel(pp_size=1)`,
    which REBUILDS the real `_PP` at world_size=1 *before* the model is constructed — silently
    discarding any earlier patch (the 2026-06-24 maiden-run OOM: every box loaded the full model).
    So we mutate the REAL group object (keeping all its real methods) from inside a wrapper around
    `ModelRunner.load_model`, which runs AFTER that re-init and right before the model is built. The
    actual torch process group still has ONE rank → no NCCL collective ever crosses a box (Cairn's
    own wire drives the forward; `world_size` here is only READ for the loader's layer partition).
    """
    group.world_size = size
    group.rank_in_group = rank
    try:
        group.ranks = list(range(size))
    except Exception:  # pragma: no cover - defensive; .ranks shape varies across sglang versions
        pass
    # is_first_rank / is_last_rank gate embed_tokens / norm+lm_head construction. If they're @property
    # (derive live from the attrs above) we're already correct; if they're plain attributes, set them.
    for attr, val in (("is_first_rank", rank == 0), ("is_last_rank", rank == size - 1)):
        if not isinstance(getattr(type(group), attr, None), property):
            try:
                setattr(group, attr, val)
            except Exception:  # pragma: no cover
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

    Path-β partial-layer load (for headline-class models that exceed per-box VRAM at full load):
      pass `cairn_pp_rank=k, cairn_pp_size=N` (or set `CAIRN_PP_{RANK,SIZE}` env vars). `load_shard`
      monkey-patches sglang's `get_pp_group()` to return a `_CairnFakePPGroup(k, N)` before
      ModelRunner construction; sglang's `make_layers` then loads ONLY layers
      `[k*L/N, (k+1)*L/N)` (the rest become weightless `PPMissingLayer`). sglang's own dist init
      still runs world_size=1 (no TCP rendezvous between Cairn boxes, no NCCL across nodes — the
      2026-06-19 invariant). `layer_range` MUST match the sglang partition; `load_shard` asserts.
    """

    _SHARED: Dict = {}   # (model, device, pp_rank, pp_size) -> ModelRunner; one per process

    def __init__(self, model: str, layer_range: LayerRange, device: str = "cuda:0",
                 quant=None, *, cairn_pp_rank: Optional[int] = None,
                 cairn_pp_size: Optional[int] = None,
                 cairn_pp_partition: Optional[list] = None) -> None:
        super().__init__(model, layer_range, device)
        self.quant = quant                       # None = the model's native dtype (don't force a quant)
        parts = device.split(":")                # tolerate "cpu"/"cuda"/"cuda:N" without crashing (L1)
        self.device_index = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        import os
        rank = cairn_pp_rank if cairn_pp_rank is not None else os.environ.get("CAIRN_PP_RANK")
        size = cairn_pp_size if cairn_pp_size is not None else os.environ.get("CAIRN_PP_SIZE")
        self.cairn_pp_rank = int(rank) if rank is not None else None
        self.cairn_pp_size = int(size) if size is not None else None
        # Optional EXPLICIT layer partition (per-stage layer counts, e.g. [15,14,14] for 43 layers).
        # Pins sglang's slice boundaries to Cairn's via SGLANG_PP_LAYER_PARTITION, instead of relying
        # on sglang's even-split default — required when Cairn's split differs (it does: front-loaded
        # vs sglang's remainder-last) and the door to memory-balanced non-uniform splits (MoE layers
        # are far heavier than the few dense layers). None ⇒ fall back to sglang's even split.
        part = cairn_pp_partition if cairn_pp_partition is not None else os.environ.get("CAIRN_PP_LAYER_PARTITION")
        if isinstance(part, str):
            part = [int(x) for x in part.split(",") if x.strip()] or None
        self.cairn_pp_partition = list(part) if part else None
        if (self.cairn_pp_rank is None) != (self.cairn_pp_size is None):
            raise ValueError("cairn_pp_rank and cairn_pp_size must be set together (or both None)")
        if self.cairn_pp_size is not None:
            if not (0 <= self.cairn_pp_rank < self.cairn_pp_size):
                raise ValueError(
                    f"cairn_pp_rank ({self.cairn_pp_rank}) must be in [0, cairn_pp_size={self.cairn_pp_size})"
                )
            if self.cairn_pp_size < 1:
                raise ValueError(f"cairn_pp_size must be ≥ 1 (got {self.cairn_pp_size})")
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

        key = (self.model, self.device, self.cairn_pp_rank, self.cairn_pp_size)
        runner = SglangNodeRuntime._SHARED.get(key)
        if runner is None:
            mem = float(os.environ.get("CAIRN_SGLANG_MEM_FRACTION", "0.8"))
            sock = socket.socket(); sock.bind(("", 0)); port = sock.getsockname()[1]; sock.close()
            sa = ServerArgs(model_path=self.model, tp_size=1, pp_size=1, mem_fraction_static=mem,
                            disable_cuda_graph=True, trust_remote_code=True)
            # Path-β layer slicing (2026-06-24 root-cause + rewrite — replaces the get_pp_group patch
            # that was silently discarded by ModelRunner's dist re-init; see _set_pp_identity docstring).
            #   (1) Pin the exact slice boundaries via SGLANG_PP_LAYER_PARTITION (so sglang's
            #       get_pp_indices matches Cairn's layer_range — incl. front-loaded / non-uniform splits).
            #   (2) Wrap ModelRunner.load_model — runs AFTER init_torch_distributed, just before the
            #       model is built — to mutate the live _PP to (k, N). The model then reads
            #       get_pp_group() = (k, N) → make_layers slices → only our layers materialize.
            # ⚠ GPU-UNVERIFIED (box torn down before re-test): confirm on next launch (a) `load_model`
            # is the post-dist-init hook point, (b) is_first/last_rank derive from the attrs we set,
            # (c) PPMissingLayer outside the slice + per-box VRAM ~slice-sized, not full.
            saved_load_model = None
            if self.cairn_pp_size is not None and self.cairn_pp_size > 1:
                k, N = self.cairn_pp_rank, self.cairn_pp_size
                if self.cairn_pp_partition is not None:
                    if len(self.cairn_pp_partition) != N:
                        raise ValueError(
                            f"cairn_pp_partition {self.cairn_pp_partition} must have cairn_pp_size={N} "
                            f"entries (one layer-count per stage)"
                        )
                    os.environ["SGLANG_PP_LAYER_PARTITION"] = ",".join(str(x) for x in self.cairn_pp_partition)
                if not hasattr(ModelRunner, "load_model"):
                    raise RuntimeError(
                        "path-β: ModelRunner has no 'load_model' to hook — sglang internals changed; "
                        "update the load_model wrapper hook point in fork/shard/sglang_node.py."
                    )
                saved_load_model = ModelRunner.load_model

                def _cairn_load_model(_mr, *a, _orig=saved_load_model, _k=k, _N=N, **kw):
                    # Runs after ModelRunner.init_torch_distributed() rebuilt the real _PP (world_size=1);
                    # set it to our (k, N) so the model-under-construction slices to our layers.
                    import sglang.srt.distributed.parallel_state as _pstate
                    grp = getattr(_pstate, "_PP", None)
                    if grp is not None:
                        _set_pp_identity(grp, _k, _N)
                    return _orig(_mr, *a, **kw)

                ModelRunner.load_model = _cairn_load_model
            try:
                runner = ModelRunner(
                    model_config=ModelConfig.from_server_args(sa), mem_fraction_static=mem,
                    gpu_id=self.device_index, tp_rank=0, tp_size=1, pp_rank=0, pp_size=1,
                    # MoE expert-parallel rank/size — REQUIRED by the sglang build in the V4-Blackwell
                    # image (added for MoE models like V4). Cairn splits by LAYER, not by expert, so
                    # each box holds ALL experts for its layer slice → no EP sharding (rank 0, size 1).
                    moe_ep_rank=0, moe_ep_size=1,
                    nccl_port=port, server_args=sa,
                )
            finally:
                if saved_load_model is not None:
                    ModelRunner.load_model = saved_load_model     # never leak the wrapper to other loads
            SglangNodeRuntime._SHARED[key] = runner
        self._runner = runner
        self._inner = runner.model.model
        n = len(self._inner.layers)
        s, e = self.layer_range.start, self.layer_range.end   # end EXCLUSIVE (LayerRange convention)
        if not (0 <= s < e <= n):
            raise ValueError(f"layer_range [{s},{e}) out of [0,{n}] for {self.model}")
        # Path-β invariant: when cairn_pp_size > 1, layer_range MUST match the sglang loader's slice.
        # Otherwise we'd try to forward through PPMissingLayer placeholders (no weights, just stubs).
        if self.cairn_pp_size is not None and self.cairn_pp_size > 1:
            from sglang.srt.distributed import get_pp_indices  # type: ignore[import-not-found]
            ps, pe = get_pp_indices(n, self.cairn_pp_rank, self.cairn_pp_size)
            if (s, e) != (ps, pe):
                raise ValueError(
                    f"layer_range [{s},{e}) does not match sglang pp partition [{ps},{pe}) "
                    f"for cairn_pp_rank={self.cairn_pp_rank} cairn_pp_size={self.cairn_pp_size}, "
                    f"num_layers={n}. Cairn's scheduler must set layer_range == get_pp_indices(...) "
                    "or the forward would hit PPMissingLayer placeholders."
                )
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

    # ---- warmup: pre-compile the flashinfer JIT kernels so the first REAL forward isn't where it lands ----
    def warmup(self, n_tokens: int = 4) -> None:
        """Run a throwaway prefill + decode to force flashinfer's one-time ~30-45s JIT kernel compile NOW,
        at load time, instead of on the first real forward. Critical for a warm SPARE: without it the
        spare's first forward IS the recovery forward, so the compile dominates MTTR (cross-box recovery
        measured 38.9s — almost all of it this compile). Uses a throwaway seq and frees it; no-op if
        unloaded. The compiled kernels persist in flashinfer's on-disk cache for this box."""
        if self._runner is None:
            return
        import torch
        w = self._inner.embed_tokens.weight                  # [vocab, H] — gives H, dtype, device
        H, dtype, device = w.shape[1], w.dtype, w.device
        seq = "__cairn_warmup__"
        try:
            if self._is_embed:                               # entry: token ids in
                self.forward(torch.arange(n_tokens, device=device).reshape(1, n_tokens), {"seq": seq, "pos": 0})
                self.forward(torch.tensor([[0]], device=device), {"seq": seq, "pos": n_tokens})
            else:                                            # mid/tail: hidden state in
                self.forward(torch.zeros(1, n_tokens, H, dtype=dtype, device=device), {"seq": seq, "pos": 0})
                self.forward(torch.zeros(1, 1, H, dtype=dtype, device=device), {"seq": seq, "pos": n_tokens})
        finally:
            self.free_seq(seq)                               # drop the throwaway KV — leaves real state untouched

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
            "kv_seqs": len(self._batches),   # seqs with KV cached on this block (renamed from _kv_seqs in the rung-2 rewrite)
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
