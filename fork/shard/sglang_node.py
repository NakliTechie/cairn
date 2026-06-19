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
    """Serve one contiguous block of layers via SGLang (rung 2/3, GPU)."""

    def __init__(self, model: str, layer_range: LayerRange, device: str = "cuda:0",
                 quant: str = "mxfp4") -> None:
        super().__init__(model, layer_range, device)
        self.quant = quant
        self.device_index = int(device.split(":")[1]) if ":" in device else 0
        self._engine = None                      # the loaded SGLang block engine
        self._kv_seqs: Dict[str, int] = {}       # seq_id -> tokens cached on THIS block

    # ---- load: pull only this block's weights to VRAM (handoff §6 rung 2) ----
    def load_shard(self) -> None:
        if not _HAS_TORCH:
            raise SglangNotAvailable(
                "torch+sglang required; install the box image (infra/) and run on a GPU"
            )
        # TODO(GPU): load layers [layer_range.start, layer_range.end) onto self.device via
        # the chosen path (A/B above). Pin the SGLang version. embedding stays on stage 0,
        # lm_head on stage N-1 (the scheduler's fit decides; passed via kv_meta/config).
        # CUDA-graph capture happens here too — this is the "tens of seconds" of recovery
        # reassign (spec §5.1), never a weight download (spares are pre-staged).
        raise NotImplementedError(
            "SglangNodeRuntime.load_shard: implement the block-subset load on a GPU "
            "(see module docstring, path A/B). This is the rung-2 integration point."
        )

    # ---- forward: one token through this block; per-seq KV stays here ----
    def forward(self, hidden_states: Any, kv_meta: Dict[str, Any]) -> Any:
        if self._engine is None:
            raise NotImplementedError(
                "SglangNodeRuntime.forward: run the block forward over hidden_states using "
                "this block's paged KV for kv_meta['seq'] at kv_meta['pos'] (spec §2/§4.4: "
                "KV strictly isolated per seq). Returns the next-stage hidden (or logits if "
                "this is the tail). Rung-2 GPU work."
            )
        seq = kv_meta["seq"]
        self._kv_seqs[seq] = self._kv_seqs.get(seq, 0) + 1
        # ... SGLang block forward here ...
        raise NotImplementedError  # pragma: no cover

    def free_seq(self, seq_id: str) -> None:
        """Drop a seq's KV pages on this block (on completion, or before replay-rebuild)."""
        self._kv_seqs.pop(seq_id, None)
        # TODO(GPU): release the seq's paged-KV blocks in the SGLang allocator.

    def kv_tokens(self, seq_id: str) -> int:
        return self._kv_seqs.get(seq_id, 0)

    # ---- heartbeat: real VRAM/liveness (concretely writable now) ----
    def heartbeat(self) -> dict:
        info: dict = {
            "loaded": self._engine is not None,
            "layers": [self.layer_range.start, self.layer_range.end],
            "device": self.device,
            "kv_seqs": len(self._kv_seqs),
        }
        if _HAS_TORCH and torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(self.device_index)
            info["vram_used_gb"] = round((total - free) / 2**30, 2)
            info["vram_total_gb"] = round(total / 2**30, 2)
            info["alive"] = True
        else:
            info["alive"] = False
            info["note"] = "no CUDA — this runtime is GPU-only (rung 2/3)"
        return info
