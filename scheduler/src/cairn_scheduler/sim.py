"""A tiny deterministic discrete-event simulation core.

The scheduler (spec §4) and recovery loop (spec §5) are *async, queue-driven* in the
real system. To test their logic without real time or real GPUs, we drive them on a
virtual clock: events are scheduled at virtual times and fired in order. Deterministic
(ties broken by insertion order) ⇒ flake-free assertions about ordering, occupancy, and
recovery timelines.
"""

from __future__ import annotations

import heapq
from typing import Callable, List, Optional, Tuple


class Sim:
    def __init__(self) -> None:
        self.now: float = 0.0
        self._heap: List[Tuple[float, int, Callable[[], None]]] = []
        self._seq: int = 0

    def schedule(self, delay: float, cb: Callable[[], None]) -> None:
        """Run `cb` after `delay` virtual time units (delay >= 0)."""
        if delay < 0:
            raise ValueError("delay must be >= 0")
        heapq.heappush(self._heap, (self.now + delay, self._seq, cb))
        self._seq += 1

    def run(self, until: Optional[float] = None) -> None:
        """Fire events until the queue drains (or virtual time passes `until`)."""
        while self._heap:
            t, _, cb = self._heap[0]
            if until is not None and t > until:
                self.now = until
                return
            heapq.heappop(self._heap)
            self.now = t
            cb()

    @property
    def empty(self) -> bool:
        return not self._heap
