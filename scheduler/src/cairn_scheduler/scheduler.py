"""The multi-stream scheduler (spec §4) — the core net-new build, in simulation.

Runs K independent streams through a pipeline of N stages to fill the bubble. Each
stage has a bounded FIFO input queue; stages process FIFO and emit downstream
(spec §4.3). Backpressure: a stage cannot push to a downstream queue that is at its
high-watermark, and admission at the entry is bounded by K_max (spec §4.3/§4.4).
Fairness: streams interleave FIFO — no head-of-line monopoly. The entry node drives
generation: a stream prefills (its prompt streams through, token-parallel) then joins
the decode rotation, where each sampled token re-enters the entry for the next step.

This is distributed continuous batching — vLLM-style, but spread *across* a pipeline
split. The mock `BlockRuntime` makes it run with zero GPU; the logic is identical to
what the real fork will drive.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Dict, List, Optional

from .runtime import BlockRuntime, sample
from .sim import Sim


@dataclass
class WorkItem:
    stream_id: str
    hidden: int
    position: int
    kind: str  # "prefill" | "decode"


@dataclass
class Stream:
    id: str
    prompt: List[int]
    max_new_tokens: int
    generated: List[int] = field(default_factory=list)
    # The durable token-history: every hidden that entered the entry node, in order
    # (prompt tokens, then each sampled token fed back). This is what recovery replays.
    entry_history: List[int] = field(default_factory=list)
    prefill_remaining: int = 0
    phase: str = "pending"  # pending → prefill → decode → done

    @property
    def done(self) -> bool:
        return len(self.generated) >= self.max_new_tokens


class _Stage:
    """One pipeline stage: a bounded queue + a block runtime + backpressure wiring."""

    def __init__(
        self,
        sim: Sim,
        runtime: BlockRuntime,
        index: int,
        service_time: float,
        high_watermark: int,
        scheduler: "Scheduler",
    ) -> None:
        self.sim = sim
        self.runtime = runtime
        self.scheduler = scheduler
        self.index = index
        self.service_time = service_time
        self.high_watermark = high_watermark
        self.queue: Deque[WorkItem] = deque()
        self.busy = False
        self.blocked_output: Optional[WorkItem] = None
        self.prev: Optional[_Stage] = None
        self.next: Optional[_Stage] = None
        self.on_tail: Optional[Callable[[WorkItem], None]] = None
        self.on_drain: Optional[Callable[[], None]] = None  # entry stage only
        self.alive = True
        # metrics
        self.busy_time = 0.0
        self.max_qlen = 0
        self.processed = 0

    def queue_full(self) -> bool:
        return self.high_watermark > 0 and len(self.queue) >= self.high_watermark

    def enqueue(self, item: WorkItem) -> None:
        self.queue.append(item)
        self.max_qlen = max(self.max_qlen, len(self.queue))
        self._try_start()

    def _try_start(self) -> None:
        if self.busy or self.blocked_output is not None or not self.queue or not self.alive:
            return
        item = self.queue.popleft()
        self.busy = True  # set BEFORE notifying upstream, so a re-entrant enqueue only queues

        def _complete() -> None:
            self.busy = False
            self.busy_time += self.service_time
            self.processed += 1
            out = self.runtime.forward(item.stream_id, item.hidden, item.position)
            self._emit(WorkItem(item.stream_id, out, item.position, item.kind))
            self._try_start()
            self.scheduler._check_idle()

        self.sim.schedule(self.service_time, _complete)
        # A slot just freed — let whoever feeds us push more (we're already busy).
        if self.prev is not None:
            self.prev._flush_blocked()
        elif self.on_drain is not None:
            self.on_drain()

    def _emit(self, item: WorkItem) -> None:
        if self.next is None:
            assert self.on_tail is not None
            self.on_tail(item)
            return
        if self.next.queue_full():
            self.blocked_output = item  # backpressure: hold until downstream drains
        else:
            self.next.enqueue(item)

    def _flush_blocked(self) -> None:
        if self.blocked_output is not None and self.next is not None and not self.next.queue_full():
            item = self.blocked_output
            self.blocked_output = None
            self.next.enqueue(item)
            self._try_start()


class Scheduler:
    def __init__(
        self,
        sim: Sim,
        runtimes: List[BlockRuntime],
        *,
        vocab_size: int,
        k_max: int,
        service_time: float = 1.0,
        high_watermark: int = 8,
    ) -> None:
        if not runtimes:
            raise ValueError("need at least one stage")
        if k_max < 1:
            raise ValueError("k_max must be >= 1")
        if vocab_size < 1:
            raise ValueError("vocab_size must be >= 1")
        if high_watermark < 1:
            # 0 disables queue_full → an unbounded per-stage queue an operator could grow without
            # bound (entry feeding never backpressures). Backpressure is the design; reject it (W2).
            raise ValueError("high_watermark must be >= 1")
        self.sim = sim
        self.vocab = vocab_size
        self.k_max = k_max
        self.stages: List[_Stage] = [
            _Stage(sim, rt, i, service_time, high_watermark, self) for i, rt in enumerate(runtimes)
        ]
        for i, st in enumerate(self.stages):
            st.prev = self.stages[i - 1] if i > 0 else None
            st.next = self.stages[i + 1] if i + 1 < len(self.stages) else None
        self.stages[-1].on_tail = self._on_token
        self.stages[0].on_drain = self._feed_entry

        self.active: Dict[str, Stream] = {}
        self.pending: Deque[Stream] = deque()
        self.finished: List[Stream] = []
        self._entry_buffer: Deque[WorkItem] = deque()
        self.max_active = 0
        # Recovery hooks (used by RecoveryManager): pause feeding to drain the pipeline on
        # an eviction warning; fire a one-shot callback when the pipeline goes idle; observe
        # each committed token.
        self.paused = False
        self._idle_cbs: List[Callable[[], None]] = []
        self.on_commit: Optional[Callable[[Stream, int], None]] = None

    @property
    def n(self) -> int:
        return len(self.stages)

    # --- submission + admission ---
    @staticmethod
    def _validate(stream: Stream) -> None:
        # Reject bad streams BEFORE they enter pending/active — a raise mid-admission used
        # to strand the stream in `active` and permanently skew admission accounting (M5).
        if not stream.prompt:
            raise ValueError(f"stream {stream.id}: prompt must be non-empty")
        if stream.max_new_tokens < 1:
            raise ValueError(f"stream {stream.id}: max_new_tokens must be >= 1")

    def submit(self, stream: Stream) -> None:
        self._validate(stream)
        self.pending.append(stream)
        self._admit()

    def submit_all(self, streams: List[Stream]) -> None:
        for s in streams:
            self._validate(s)
        self.pending.extend(streams)
        self._admit()

    def _admit(self) -> None:
        # Admit up to K_max (spec §4.3). New streams prefill, then join decode.
        while len(self.active) < self.k_max and self.pending:
            s = self.pending.popleft()
            self.active[s.id] = s
            self.max_active = max(self.max_active, len(self.active))
            self._start_prefill(s)

    def _start_prefill(self, s: Stream) -> None:
        s.phase = "prefill"
        s.prefill_remaining = len(s.prompt)
        s.entry_history = list(s.prompt)
        for pos, h in enumerate(s.prompt):
            self._entry_buffer.append(WorkItem(s.id, h, pos, "prefill"))
        self._feed_entry()

    def _feed_entry(self) -> None:
        # Entry-side flow control: feed stage 0 only while it has room (bounds its queue
        # like every other stage; admission is bounded by K_max, not by dumping prefill).
        if self.paused:
            return  # draining for recovery — hold all new traversals at the entry
        entry = self.stages[0]
        while self._entry_buffer and not entry.queue_full():
            entry.enqueue(self._entry_buffer.popleft())

    # --- the entry node driving generation ---
    def _on_token(self, item: WorkItem) -> None:
        s = self.active.get(item.stream_id)
        if s is None:
            return  # stream finished / dropped
        if item.kind == "prefill":
            s.prefill_remaining -= 1
            if s.prefill_remaining > 0:
                return  # intermediate prompt positions: only their KV matters, output discarded
            # last prompt position → sample the first generated token
        tok = sample(item.hidden, self.vocab)
        s.generated.append(tok)
        if self.on_commit is not None:
            self.on_commit(s, len(s.generated))
        if s.done:
            self._finish(s)
            return
        s.phase = "decode"
        s.entry_history.append(tok)
        self._entry_buffer.append(WorkItem(s.id, tok, len(s.entry_history) - 1, "decode"))
        self._feed_entry()

    def _finish(self, s: Stream) -> None:
        s.phase = "done"
        for st in self.stages:
            st.runtime.free_stream(s.id)
        self.active.pop(s.id, None)
        self.finished.append(s)
        self._admit()  # a slot freed — admit the next pending stream

    # --- recovery / drain support ---
    def pipeline_idle(self) -> bool:
        """No stage busy, all queues empty, nothing held by backpressure. With feeding
        paused, this means the in-flight work has drained to a token boundary."""
        return all(
            not st.busy and not st.queue and st.blocked_output is None for st in self.stages
        )

    def call_when_idle(self, cb: Callable[[], None]) -> None:
        """Fire `cb` once, as soon as the pipeline is idle (now, or after it drains). Queued,
        so a second eviction in the same drain window doesn't drop the first's callback (M3)."""
        self._idle_cbs.append(cb)
        self._check_idle()

    def _check_idle(self) -> None:
        if self._idle_cbs and self.pipeline_idle():
            cbs = self._idle_cbs
            self._idle_cbs = []
            for cb in cbs:
                cb()

    def resume_feeding(self) -> None:
        self.paused = False
        self._feed_entry()

    # --- metrics ---
    def occupancy(self) -> List[float]:
        t = self.sim.now or 1.0
        return [st.busy_time / t for st in self.stages]

    def avg_occupancy(self) -> float:
        occ = self.occupancy()
        return sum(occ) / len(occ) if occ else 0.0

    def max_queue_len(self) -> int:
        return max((st.max_qlen for st in self.stages), default=0)
