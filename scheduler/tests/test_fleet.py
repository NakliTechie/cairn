import pytest

from cairn_scheduler.domain import NodeState
from cairn_scheduler.fleet import FleetError, FleetState


def _activate(fleet, node_id):
    for st in (NodeState.STAGING, NodeState.WARM, NodeState.LOADING, NodeState.ACTIVE):
        fleet.set_state(node_id, st)


def _three_node_pipeline():
    f = FleetState()
    f.register_node("n0", 0, 0, 11, "v1")
    f.register_node("n1", 1, 12, 23, "v1")
    f.register_node("n2", 2, 24, 35, "v1")
    for nid in ("n0", "n1", "n2"):
        _activate(f, nid)
    f.set_topology(["n0", "n1", "n2"])
    return f


def test_register_and_pipeline_order():
    f = _three_node_pipeline()
    assert [n.node_id for n in f.pipeline()] == ["n0", "n1", "n2"]
    assert f.is_complete_pipeline()


def test_pipeline_gap_detected():
    f = FleetState()
    f.register_node("n0", 0, 0, 11, "v1")
    f.register_node("n1", 1, 13, 23, "v1")  # gap: 12 missing
    for nid in ("n0", "n1"):
        _activate(f, nid)
    f.set_topology(["n0", "n1"])
    assert not f.is_complete_pipeline()


def test_illegal_state_transition_raises():
    f = FleetState()
    f.register_node("n0", 0, 0, 11, "v1")
    with pytest.raises(FleetError):
        f.set_state("n0", NodeState.ACTIVE)  # PROVISIONING → ACTIVE is not legal


def test_health_stale_detection():
    f = _three_node_pipeline()
    for nid in ("n0", "n1", "n2"):
        f.heartbeat(nid, 100.0)
    f.heartbeat("n1", 50.0)  # n1 went quiet
    assert f.stale_nodes(now=110.0, timeout=30.0) == ["n1"]
    assert f.stale_nodes(now=110.0, timeout=120.0) == []


def test_token_history_record_and_drop():
    f = FleetState()
    f.record_tokens("s", [1, 2, 3])
    f.record_tokens("s", [4, 5])
    assert f.history_of("s") == [1, 2, 3, 4, 5]
    f.drop_stream("s")
    assert f.history_of("s") == []


def test_select_recovery_policy():
    f = _three_node_pipeline()
    assert f.select_recovery("n1", warm_spares=1, spare_version="v1")[0] == "reassign"
    assert f.select_recovery("n1", warm_spares=0, spare_version="v1")[0] == "rebuild"
    assert f.select_recovery("n1", warm_spares=1, spare_version="v2")[0] == "rebuild"  # skew


def test_eviction_warning_sets_draining():
    f = _three_node_pipeline()
    f.note_eviction_warning("n1")
    assert f.nodes["n1"].state == NodeState.DRAINING
