"""Fleet state — the Durable-Objects logic (spec §7), in plain Python so it is testable
and portable to a CF Durable Object later.

Holds: the **registry** (who's in the fleet, which block, version/quant), the **topology**
(pipeline order), **health** (heartbeats → the source the recovery loop reads), and the
**durable stream token-history** (authoritative token-IDs per in-flight stream, mirrored
from the entry node so even a dead entry node is recoverable — invariant #2). It also
selects the `RecoveryPolicy` for the drain/retry/reassign loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .domain import NodeState, can_transition


class FleetError(ValueError):
    pass


@dataclass
class NodeInfo:
    node_id: str
    stage: int
    layer_start: int
    layer_end: int
    version: str          # weight-version + quant tag — drives skew detection (spec §5.1)
    state: NodeState = NodeState.PROVISIONING
    last_heartbeat: float = 0.0


class FleetState:
    def __init__(self) -> None:
        self.nodes: Dict[str, NodeInfo] = {}
        self.topology: List[str] = []                # node_ids in pipeline order
        self.token_history: Dict[str, List[int]] = {}

    # --- registry ---
    def register_node(self, node_id, stage, layer_start, layer_end, version, t=0.0) -> NodeInfo:
        if node_id in self.nodes:
            raise FleetError(f"node {node_id} already registered")
        info = NodeInfo(node_id, stage, layer_start, layer_end, version, NodeState.PROVISIONING, t)
        self.nodes[node_id] = info
        return info

    def set_state(self, node_id: str, state: NodeState) -> None:
        info = self._node(node_id)
        if not can_transition(info.state, state):
            raise FleetError(f"illegal transition {info.state.value} → {state.value} for {node_id}")
        info.state = state

    def _node(self, node_id: str) -> NodeInfo:
        if node_id not in self.nodes:
            raise FleetError(f"unknown node {node_id}")
        return self.nodes[node_id]

    # --- topology ---
    def set_topology(self, ordered_node_ids: List[str]) -> None:
        for nid in ordered_node_ids:
            self._node(nid)
        self.topology = list(ordered_node_ids)

    def pipeline(self) -> List[NodeInfo]:
        return [self.nodes[nid] for nid in self.topology]

    def is_complete_pipeline(self) -> bool:
        """Topology covers a contiguous layer range with every node ACTIVE."""
        nodes = self.pipeline()
        if not nodes:
            return False
        if any(n.state != NodeState.ACTIVE for n in nodes):
            return False
        expect = nodes[0].layer_start
        for n in nodes:
            if n.layer_start != expect:
                return False
            expect = n.layer_end + 1
        return True

    # --- health ---
    def heartbeat(self, node_id: str, t: float) -> None:
        self._node(node_id).last_heartbeat = t

    def stale_nodes(self, now: float, timeout: float) -> List[str]:
        """Nodes whose heartbeat has aged past `timeout` — the hard-crash signal that the
        supervised edges raise (spec §6) and the recovery loop reads."""
        return [
            nid for nid, n in self.nodes.items()
            if n.state == NodeState.ACTIVE and (now - n.last_heartbeat) > timeout
        ]

    # --- durable stream token-history (invariant #2) ---
    def record_tokens(self, stream_id: str, token_ids: List[int]) -> None:
        self.token_history.setdefault(stream_id, []).extend(token_ids)

    def history_of(self, stream_id: str) -> List[int]:
        return list(self.token_history.get(stream_id, []))

    def drop_stream(self, stream_id: str) -> None:
        self.token_history.pop(stream_id, None)  # freed on completion (spec §8 retention)

    # --- the drain/retry/reassign loop's policy selection (spec §5.1) ---
    def select_recovery(self, node_id: str, warm_spares: int, spare_version: str) -> Tuple[str, str]:
        node = self._node(node_id)
        if warm_spares <= 0:
            return "rebuild", "no warm spare available"
        if node.version != spare_version:
            return "rebuild", f"version skew ({node.version} != {spare_version}); activations would be garbage"
        return "reassign", "warm spare available, versions match"

    def note_eviction_warning(self, node_id: str) -> None:
        self.set_state(node_id, NodeState.DRAINING)
