"""The per-node block-runtime seam (spec §2) + a no-GPU mock (handoff §6, rung 1).

`BlockRuntime` is the contract every stage answers to: load one contiguous block of
layers, run `forward(hidden, kv_meta) → hidden`, hold per-node KV for its block across
all streams. The real implementation is the **stripped Shard fork (SGLang)** — it
plugs in here at rungs 2–3. `MockBlockRuntime` is the rung-1 stand-in: it simulates the
*plumbing and state-flow* (per-stream KV growth + isolation, replay-reconstructs-state,
simulated per-stage latency) with deterministic integer "math" — NOT real numerics.

Why deterministic ints: the rung-1 gates are about orchestration correctness —
cross-stream KV isolation (spec §4.4) and replay-rebuild (spec §5.3) — not floating
point. Integer state makes "token-for-token identical" an exact, flake-free assertion.
"""

from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional, Protocol, runtime_checkable

_MASK = (1 << 64) - 1


def _mix(a: int, b: int) -> int:
    """Deterministic splitmix64-style 64-bit mix. Stands in for a block's forward pass."""
    x = (a + 0x9E3779B97F4A7C15 + ((b * 0xBF58476D1CE4E5B9) & _MASK)) & _MASK
    x ^= x >> 30
    x = (x * 0xBF58476D1CE4E5B9) & _MASK
    x ^= x >> 27
    x = (x * 0x94D049BB133111EB) & _MASK
    x ^= x >> 31
    return x & _MASK


@runtime_checkable
class BlockRuntime(Protocol):
    """One pipeline stage: a contiguous block of layers + its per-stream KV."""

    stage: int
    layer_start: int
    layer_end: int

    def load(self) -> None:
        """VRAM-load + (real impl) CUDA-graph capture. WARM → LOADING → ACTIVE."""
        ...

    def forward(self, stream_id: str, hidden: int, position: int) -> int:
        """Advance one token through this block; append to the stream's KV. Returns the
        block's output hidden, fed to the next stage (or sampled, on the tail)."""
        ...

    def kv_len(self, stream_id: str) -> int:
        """Number of tokens currently cached for `stream_id` on this block."""
        ...

    def free_stream(self, stream_id: str) -> None:
        """Drop a stream's KV (on completion, or before a replay-rebuild)."""
        ...


class MockBlockRuntime:
    """No-GPU stand-in for one stage. Deterministic, fast, KV-isolating."""

    def __init__(
        self,
        stage: int,
        layer_start: int,
        layer_end: int,
        *,
        latency_s: float = 0.0,
        sleep: Optional[Callable[[float], None]] = None,
        seed: int = 0xC0FFEE,
    ) -> None:
        self.stage = stage
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.latency_s = latency_s
        self._sleep = sleep
        self._seed = seed
        self.loaded = False
        # Per-stream, per-layer KV: stream -> {layer -> folded key/value history}.
        # Keyed per-LAYER (not per-stage) so the per-token output is INDEPENDENT of where
        # stage boundaries fall → a split pipeline is token-for-token identical to a
        # single reference run (the spec §9 v1.0 correctness gate, modelled in sim).
        self._kv: Dict[str, Dict[int, int]] = {}
        self._kv_len: Dict[str, int] = {}

    @property
    def layers(self) -> range:
        # Empty range for a lm_head-only tail stage (layer_end < layer_start).
        return range(self.layer_start, self.layer_end + 1)

    @property
    def num_layers(self) -> int:
        return max(0, self.layer_end - self.layer_start + 1)

    def load(self) -> None:
        self.loaded = True

    def forward(self, stream_id: str, hidden: int, position: int) -> int:
        if self.latency_s and self._sleep is not None:
            self._sleep(self.latency_s)
        if self.num_layers == 0:
            return hidden  # lm_head-only tail stage: sampling is applied by the caller
        layer_kv = self._kv.setdefault(stream_id, {})
        h = hidden
        for layer in self.layers:
            prev = layer_kv.get(layer, self._seed)  # this layer's KV history for the stream
            kin = h                                  # hidden entering this layer (its K/V source)
            h = _mix(kin ^ prev, layer)              # output ← input + KV history + weights(layer)
            layer_kv[layer] = _mix(prev, kin)        # append this token's K/V to the layer's cache
        self._kv_len[stream_id] = self._kv_len.get(stream_id, 0) + 1
        return h

    def kv_len(self, stream_id: str) -> int:
        return self._kv_len.get(stream_id, 0)

    def free_stream(self, stream_id: str) -> None:
        self._kv.pop(stream_id, None)
        self._kv_len.pop(stream_id, None)

    def has_stream(self, stream_id: str) -> bool:
        return stream_id in self._kv


def build_mock_pipeline(fit_result, **kwargs) -> List[MockBlockRuntime]:
    """Materialise a list of MockBlockRuntimes from a FitResult — the rung-1 fleet."""
    return [
        MockBlockRuntime(a.stage, a.layer_start, a.layer_end, **kwargs)
        for a in fit_result.assignments
    ]


def chain_forward(stages: Iterable[BlockRuntime], stream_id: str, hidden: int, position: int) -> int:
    """Run one token through the whole pipeline (stage 0 → … → tail). Returns the tail
    output hidden — what the entry node samples to get the next token."""
    out = hidden
    for stage in stages:
        out = stage.forward(stream_id, out, position)
    return out


_LM_HEAD_SALT = 0xA5A5A5A5A5A5A5A5


def sample(hidden: int, vocab_size: Optional[int] = None) -> int:
    """The lm_head + greedy sample, applied ONCE after the tail stage. Kept external to
    the blocks so it is identical for a reference run and a split run (partition-
    invariance). Deterministic ⇒ greedy decode is exactly reproducible."""
    tok = _mix(hidden, _LM_HEAD_SALT)
    return tok % vocab_size if vocab_size else tok
