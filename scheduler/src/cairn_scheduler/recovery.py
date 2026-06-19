"""Recovery orchestration (spec §5) — in simulation.

`RecoveryPolicy`: **reassign** (default hot path, warm spare) and **rebuild** (fallback,
when no spare / version-skew). KV is never migrated — it is **rebuilt by replaying the
durable token-history** over the dead block (spec §5.3). The §5.4 node state machine
(WARM→LOADING→ACTIVE, ACTIVE→DRAINING→dead) is tracked.

Two halves at two timescales (handoff §4): the warm spare takes the block in seconds
(modelled as `load_s`); SkyPilot backfilling the slot is out of scope for the sim.

Integrated path = the eviction-warning **drain** (§5.4): the node stays alive ~2 min,
the gateway stops feeding new tokens, in-flight work drains to a token boundary, then
the spare is stitched in. The hard-crash "re-drive the lost token from history" property
is shown directly by `replay_rebuild_kv` (a rebuild to the committed boundary).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .domain import NodeState
from .runtime import BlockRuntime, MockBlockRuntime
from .scheduler import Scheduler
from .sim import Sim


def _fresh_like(rt: BlockRuntime) -> MockBlockRuntime:
    """A pre-staged warm spare for the same block — VRAM-cold, no KV yet."""
    return MockBlockRuntime(rt.stage, rt.layer_start, rt.layer_end)


def replay_rebuild_kv(
    upstream_runtimes: Sequence[BlockRuntime],
    target: BlockRuntime,
    stream_id: str,
    entry_history: Sequence[int],
    target_len: int,
) -> None:
    """Rebuild `target`'s KV for one stream to `target_len` tokens by replaying the
    committed entry-history through **fresh copies** of the upstream blocks — so the live
    upstream caches are untouched (spec §5.3: "downstream caches are untouched; re-
    streaming through them would double-append and corrupt them"). Only `target`'s cache
    is populated. This is the concrete form of "recovery is replay, not migrate."
    """
    temp_upstream = [_fresh_like(rt) for rt in upstream_runtimes]
    for pos, h in enumerate(entry_history[:target_len]):
        x = h
        for tu in temp_upstream:
            x = tu.forward(stream_id, x, pos)
        target.forward(stream_id, x, pos)


@dataclass
class RecoveryEvent:
    t_warning: float
    stage: int
    policy: str            # "reassign" | "rebuild"
    affected_streams: int
    replay_tokens: int
    load_s: float
    reprefill_s: float
    t_resume: float


class RecoveryManager:
    def __init__(
        self,
        sim: Sim,
        scheduler: Scheduler,
        *,
        warm_spares: int = 1,
        load_s: float = 20.0,
        reprefill_per_token: float = 0.01,
    ) -> None:
        self.sim = sim
        self.sched = scheduler
        self.warm_spares = warm_spares
        self.load_s = load_s
        self.reprefill_per_token = reprefill_per_token
        self.timeline: List[RecoveryEvent] = []
        self.node_state: Dict[int, NodeState] = {
            i: NodeState.ACTIVE for i in range(scheduler.n)
        }
        self._armed: Optional[dict] = None

    def arm(self, stage_index: int, *, after_stream: str, after_count: int,
            version_skew: bool = False) -> None:
        """Trigger an eviction of `stage_index` once `after_stream` commits `after_count`
        tokens — a deterministic mid-decode interruption for the gate test."""
        self._armed = {
            "stage": stage_index, "stream": after_stream,
            "count": after_count, "skew": version_skew, "fired": False,
        }
        self.sched.on_commit = self._maybe_fire

    def _maybe_fire(self, stream, n_committed: int) -> None:
        a = self._armed
        if a and not a["fired"] and stream.id == a["stream"] and n_committed >= a["count"]:
            a["fired"] = True
            self.evict(a["stage"], version_skew=a["skew"])

    def evict(self, stage_index: int, *, version_skew: bool = False) -> None:
        """Eviction warning → DRAINING; stop feeding new tokens; recover once drained."""
        t_warn = self.sim.now
        self.node_state[stage_index] = NodeState.DRAINING
        self.sched.paused = True
        self.sched.call_when_idle(lambda: self._recover(stage_index, t_warn, version_skew))

    def _recover(self, i: int, t_warn: float, version_skew: bool) -> None:
        affected = list(self.sched.active.keys())
        has_spare = self.warm_spares > 0
        # Reassign needs a warm spare AND no weight-version/quant skew (cross-version
        # activations are garbage → must rebuild) — spec §5.1.
        policy = "reassign" if (has_spare and not version_skew) else "rebuild"

        self.node_state[i] = NodeState.LOADING  # spare: VRAM-load + graph capture
        old_stage = self.sched.stages[i]
        new_rt = _fresh_like(old_stage.runtime)
        upstream = [st.runtime for st in self.sched.stages[:i]]
        survivor = self.sched.stages[i + 1] if i + 1 < self.sched.n else self.sched.stages[i - 1]

        replay_tokens = 0
        for sid in affected:
            target_len = survivor.runtime.kv_len(sid)  # rebuild to match surviving stages
            replay_rebuild_kv(upstream, new_rt, sid, self.sched.active[sid].entry_history, target_len)
            replay_tokens += target_len

        # Re-stitch: the _Stage object stays, edges already point to it — swap the runtime.
        old_stage.runtime = new_rt
        old_stage.alive = True
        if policy == "reassign":
            self.warm_spares -= 1

        load_s = self.load_s
        reprefill_s = replay_tokens * self.reprefill_per_token

        def _resume() -> None:
            self.node_state[i] = NodeState.ACTIVE
            self.sched.resume_feeding()

        self.sim.schedule(load_s + reprefill_s, _resume)  # model the recovery gap
        self.timeline.append(
            RecoveryEvent(
                t_warning=t_warn, stage=i, policy=policy, affected_streams=len(affected),
                replay_tokens=replay_tokens, load_s=load_s, reprefill_s=reprefill_s,
                t_resume=self.sim.now + load_s + reprefill_s,
            )
        )
