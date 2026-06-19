from cairn_scheduler import fit
from cairn_scheduler.recovery import RecoveryManager, replay_rebuild_kv
from cairn_scheduler.runtime import MockBlockRuntime, build_mock_pipeline, chain_forward
from cairn_scheduler.scheduler import Scheduler, Stream
from cairn_scheduler.sim import Sim


def _fit(cfg):
    return fit(cfg, target_k=8, context_len=4096)


def test_replay_rebuild_block_resumes_identically(gpt_oss_cfg):
    """spec §5.3: rebuild ONLY the dead block by replaying the token-history; downstream
    untouched; subsequent forwards are bit-identical to a no-crash run."""
    r = _fit(gpt_oss_cfg)
    assert r.n >= 3
    sid = "s"
    seq = [(j * 2654435761) & 0xFFFF for j in range(1, 21)]  # 20 fixed inputs

    ref = build_mock_pipeline(r)
    ref_outs = [chain_forward(ref, sid, h, pos) for pos, h in enumerate(seq)]

    test = build_mock_pipeline(r)
    T = 10
    for pos in range(T):
        chain_forward(test, sid, seq[pos], pos)  # prime all stages' KV to T tokens

    i = r.n // 2  # a middle block dies
    others_before = [rt for j, rt in enumerate(test) if j != i]
    new_rt = MockBlockRuntime(test[i].stage, test[i].layer_start, test[i].layer_end)
    replay_rebuild_kv([test[j] for j in range(i)], new_rt, sid, seq, T)
    test[i] = new_rt  # re-stitch the spare in place

    for pos in range(T, len(seq)):
        assert chain_forward(test, sid, seq[pos], pos) == ref_outs[pos]  # uncorrupted
    assert [rt for j, rt in enumerate(test) if j != i] == others_before  # only block i rebuilt


def _run(cfg, specs, *, k_max, evict_stage=None, after="s0", count=0,
         warm_spares=1, version_skew=False):
    sim = Sim()
    sched = Scheduler(sim, build_mock_pipeline(_fit(cfg)), vocab_size=cfg.vocab_size, k_max=k_max)
    rm = RecoveryManager(sim, sched, warm_spares=warm_spares)
    if evict_stage is not None:
        rm.arm(evict_stage, after_stream=after, after_count=count, version_skew=version_skew)
    sched.submit_all([Stream(id=i, prompt=list(p), max_new_tokens=m) for i, p, m in specs])
    sim.run()
    return sched, rm, {s.id: s.generated for s in sched.finished}


def _specs():
    return [(f"s{i}", [i + 1, i + 2, i + 3], 15) for i in range(4)]


def test_reassign_keeps_output_uncorrupted(gpt_oss_cfg):
    """v1.0 induced-interruption gate (in sim): kill a node mid-decode → reassign to the
    warm spare → resume; every stream completes with output identical to the no-crash run."""
    n = _fit(gpt_oss_cfg).n
    _, _, ref = _run(gpt_oss_cfg, _specs(), k_max=4)
    sched, rm, got = _run(gpt_oss_cfg, _specs(), k_max=4, evict_stage=n // 2, after="s0", count=5)

    assert len(sched.finished) == 4
    assert got == ref                       # uncorrupted across the interruption
    assert len(rm.timeline) == 1
    assert rm.timeline[0].policy == "reassign"
    assert rm.warm_spares == 0              # the warm spare was consumed


def test_rebuild_fallback_when_no_spare(gpt_oss_cfg):
    n = _fit(gpt_oss_cfg).n
    _, _, ref = _run(gpt_oss_cfg, _specs(), k_max=4)
    sched, rm, got = _run(gpt_oss_cfg, _specs(), k_max=4, evict_stage=n // 2, count=5, warm_spares=0)
    assert got == ref
    assert rm.timeline[0].policy == "rebuild"  # no warm spare → rebuild fallback (spec §5.1)


def test_version_skew_forces_rebuild(gpt_oss_cfg):
    n = _fit(gpt_oss_cfg).n
    _, _, ref = _run(gpt_oss_cfg, _specs(), k_max=4)
    sched, rm, got = _run(
        gpt_oss_cfg, _specs(), k_max=4, evict_stage=n // 2, count=5, warm_spares=1, version_skew=True
    )
    assert got == ref
    assert rm.timeline[0].policy == "rebuild"  # cross-version activations are garbage → rebuild
    assert rm.warm_spares == 1                  # spare NOT consumed by a rebuild


def test_recovery_timeline_logged(gpt_oss_cfg):
    from cairn_scheduler.domain import NodeState

    n = _fit(gpt_oss_cfg).n
    sched, rm, _ = _run(gpt_oss_cfg, _specs(), k_max=4, evict_stage=n // 2, count=5)
    ev = rm.timeline[0]
    assert ev.stage == n // 2
    assert ev.replay_tokens > 0
    assert ev.t_resume > ev.t_warning           # recovery took virtual time (the gap)
    assert rm.node_state[n // 2] == NodeState.ACTIVE  # node back to ACTIVE after resume
