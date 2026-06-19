"""Core domain types: the node state machine (spec §5.4) and fit outputs (spec §3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


class NodeState(str, Enum):
    """Node lifecycle — spec §5.4, verbatim.

        provisioning → staging (weights→disk) → warm (disk-staged, VRAM-cold)
           → loading (VRAM-load + graph capture) → active (serving block)
           → draining (eviction warning) → dead

    A *warm spare* sits at WARM (disk-staged, VRAM-cold). Reassign-recovery is the
    WARM → LOADING → ACTIVE transition for one block (tens of seconds, no download).
    """

    PROVISIONING = "provisioning"
    STAGING = "staging"
    WARM = "warm"
    LOADING = "loading"
    ACTIVE = "active"
    DRAINING = "draining"
    DEAD = "dead"


# Legal transitions (spec §5.4). DEAD is terminal; ACTIVE→DEAD covers a hard crash
# (edge-supervision timeout) with no DRAINING warning.
_TRANSITIONS = {
    NodeState.PROVISIONING: {NodeState.STAGING, NodeState.DEAD},
    NodeState.STAGING: {NodeState.WARM, NodeState.DEAD},
    NodeState.WARM: {NodeState.LOADING, NodeState.DRAINING, NodeState.DEAD},
    NodeState.LOADING: {NodeState.ACTIVE, NodeState.DEAD},
    NodeState.ACTIVE: {NodeState.DRAINING, NodeState.DEAD},
    NodeState.DRAINING: {NodeState.DEAD},
    NodeState.DEAD: set(),
}


def can_transition(src: NodeState, dst: NodeState) -> bool:
    """True if src → dst is a legal node-state-machine edge (spec §5.4)."""
    return dst in _TRANSITIONS[src]


@dataclass(frozen=True)
class BlockAssignment:
    """One contiguous block of layers placed on one node (spec §3, invariant #1).

    `layer_start`..`layer_end` is inclusive. A node may also pin the embedding
    (stage 0) and/or the lm_head (stage N−1). `est_vram_bytes` is the fit's estimate
    of this stage's footprint = block weights + (embedding/lm_head) + reserved KV
    headroom for the target K + activation buffer.
    """

    stage: int
    layer_start: int
    layer_end: int
    holds_embedding: bool
    holds_lm_head: bool
    est_vram_bytes: int

    @property
    def num_layers(self) -> int:
        # A pure lm_head tail stage can carry zero layers.
        if self.layer_end < self.layer_start:
            return 0
        return self.layer_end - self.layer_start + 1


@dataclass(frozen=True)
class FitResult:
    """Output of the §3 fit: the pipeline as a list of stages, plus the headroom math."""

    model_name: str
    assignments: List[BlockAssignment]
    target_k: int
    max_context: int
    usable_vram_bytes: int
    # The single most-loaded stage's est VRAM — the pipeline runs at the speed of its
    # slowest/fullest stage, so the fit minimises this (spec §3).
    max_stage_vram_bytes: int = field(default=0)

    @property
    def n(self) -> int:
        """N = number of pipeline stages (nodes). The blast radius is 1/N."""
        return len(self.assignments)
