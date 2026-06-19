"""Adapter: the real Shard block runtime → Cairn's `BlockRuntime` seam.

`cairn_scheduler.runtime.BlockRuntime` is the contract the scheduler drives; the no-GPU
`MockBlockRuntime` is the rung-1 implementation. This is the rung-2/3 implementation:
`ShardBlockRuntime` wraps `shard.node.NodeRuntime` (the SGLang block-serving wrap) so the
SAME scheduler/recovery code runs the real fleet — swapping sim↔real is a
`build_mock_pipeline()` → `build_shard_pipeline()` change, nothing else.

GPU boundary: `load()`/`forward()` delegate to `NodeRuntime`, which wraps SGLang and needs
CUDA — on CPU they raise NotImplementedError (rung-2 work, needs a GPU). The seam shape and
KV bookkeeping are real now and verified against the protocol on CPU.
"""

from __future__ import annotations

from typing import Dict, List, Type

from shard.node import LayerRange, NodeRuntime


class ShardBlockRuntime:
    """One real pipeline stage: a contiguous block of layers served by SGLang.

    `runtime_cls` selects the backing NodeRuntime: the base contract (default — its
    stubs raise NotImplementedError, so this is a CPU-safe no-op) or, on a GPU,
    `shard.sglang_node.SglangNodeRuntime`."""

    def __init__(self, stage: int, layer_start: int, layer_end: int, model: str,
                 device: str = "cuda:0", runtime_cls: Type[NodeRuntime] = NodeRuntime) -> None:
        self.stage = stage
        self.layer_start = layer_start            # inclusive (Cairn convention)
        self.layer_end = layer_end                # inclusive
        # NodeRuntime.LayerRange.end is EXCLUSIVE — convert.
        self._node = runtime_cls(model, LayerRange(layer_start, layer_end + 1), device)
        self._kv_len: Dict[str, int] = {}

    def load(self) -> None:
        self._node.load_shard()                   # pull this block's weights → VRAM (GPU, rung 2)

    def forward(self, stream_id: str, hidden, position: int):
        # SGLang block forward; kv_meta carries the seq id + position so the node manages
        # its own KV-cache for this block across all streams (spec §2).
        out = self._node.forward(hidden, {"seq": stream_id, "pos": position})  # GPU, rung 2
        self._kv_len[stream_id] = self._kv_len.get(stream_id, 0) + 1
        return out

    def kv_len(self, stream_id: str) -> int:
        return self._kv_len.get(stream_id, 0)

    def free_stream(self, stream_id: str) -> None:
        self._kv_len.pop(stream_id, None)
        # Release the seq's KV on the backing runtime if it supports it (SglangNodeRuntime
        # has free_seq; the base NodeRuntime contract does not) — S11.
        free_seq = getattr(self._node, "free_seq", None)
        if callable(free_seq):
            free_seq(stream_id)


def build_shard_pipeline(fit_result, model: str, device: str = "cuda:0",
                         runtime_cls: Type[NodeRuntime] = NodeRuntime) -> List[ShardBlockRuntime]:
    """Materialise real ShardBlockRuntimes from a FitResult — the GPU counterpart of
    `cairn_scheduler.runtime.build_mock_pipeline`. The same FitResult drives both. On a
    GPU pass `runtime_cls=SglangNodeRuntime` (from shard.sglang_node)."""
    return [
        ShardBlockRuntime(a.stage, a.layer_start, a.layer_end, model, device, runtime_cls)
        for a in fit_result.assignments
    ]
