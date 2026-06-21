import pytest

from cairn_scheduler import fit
from cairn_scheduler.runtime import MockBlockRuntime, build_mock_pipeline
from cairn_scheduler.scheduler import Scheduler, Stream
from cairn_scheduler.sim import Sim


def _run(make_runtimes, specs, vocab, *, k_max, high_watermark=8):
    """Run streams to completion. `specs` = [(id, prompt, max_new), ...]."""
    sim = Sim()
    sched = Scheduler(
        sim, make_runtimes(), vocab_size=vocab, k_max=k_max, high_watermark=high_watermark
    )
    sched.submit_all([Stream(id=i, prompt=list(p), max_new_tokens=m) for i, p, m in specs])
    sim.run()
    return sched, {s.id: s.generated for s in sched.finished}


def test_single_stream_generates(gpt_oss_cfg):
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    sched, gen = _run(lambda: build_mock_pipeline(r), [("s", [5, 6, 7], 8)], gpt_oss_cfg.vocab_size, k_max=4)
    assert len(gen["s"]) == 8
    assert len(sched.finished) == 1


def test_split_matches_single_node_reference(gpt_oss_cfg):
    """spec §9 v1.0 correctness gate (in sim, through the full scheduler): the split
    pipeline decodes token-for-token identically to a single-node reference."""
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    L = gpt_oss_cfg.num_layers
    vocab = gpt_oss_cfg.vocab_size
    spec = [("s", [11, 22, 33, 44], 16)]

    _, ref = _run(lambda: [MockBlockRuntime(0, 0, L - 1)], spec, vocab, k_max=1)
    sched, split = _run(lambda: build_mock_pipeline(r), spec, vocab, k_max=4)

    assert sched.n >= 2  # the gate is only meaningful if it actually split
    assert split["s"] == ref["s"]


def test_k_streams_match_solo(gpt_oss_cfg):
    """Concurrency must not change any stream's output (KV isolation at scheduler level)."""
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    vocab = gpt_oss_cfg.vocab_size
    specs = [(f"s{i}", [i + 1, i + 2, i + 3], 12) for i in range(5)]

    _, together = _run(lambda: build_mock_pipeline(r), specs, vocab, k_max=5)
    for sid, prompt, mn in specs:
        _, solo = _run(lambda: build_mock_pipeline(r), [(sid, prompt, mn)], vocab, k_max=1)
        assert together[sid] == solo[sid]


def test_admission_respects_k_max(gpt_oss_cfg):
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    specs = [(f"s{i}", [1, 2], 6) for i in range(8)]
    sched, gen = _run(lambda: build_mock_pipeline(r), specs, gpt_oss_cfg.vocab_size, k_max=3)
    assert sched.max_active <= 3
    assert len(sched.finished) == 8  # all still complete, just rotated through


def test_backpressure_bounds_queues(gpt_oss_cfg):
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    specs = [(f"s{i}", [1, 2, 3], 10) for i in range(6)]
    sched, _ = _run(lambda: build_mock_pipeline(r), specs, gpt_oss_cfg.vocab_size, k_max=6, high_watermark=4)
    assert sched.max_queue_len() <= 4  # no stage queue ever exceeds the high-watermark


def test_rejects_bad_streams(gpt_oss_cfg):
    """M1/M5: empty prompt or max_new_tokens<1 is rejected at submit, before the stream
    can enter `active` (no admission-accounting pollution)."""
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    sim = Sim()
    sched = Scheduler(sim, build_mock_pipeline(r), vocab_size=gpt_oss_cfg.vocab_size, k_max=4)
    with pytest.raises(ValueError):
        sched.submit(Stream(id="empty", prompt=[], max_new_tokens=4))
    with pytest.raises(ValueError):
        sched.submit(Stream(id="zero", prompt=[1, 2], max_new_tokens=0))
    assert len(sched.active) == 0 and sched.max_active == 0  # nothing polluted admission


def test_rejects_bad_config(gpt_oss_cfg):
    """M2: k_max<1 / vocab_size<1 — and W2: high_watermark<1 — fail loudly instead of silently
    admitting nothing / running an unbounded (never-backpressuring) queue."""
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    with pytest.raises(ValueError):
        Scheduler(Sim(), build_mock_pipeline(r), vocab_size=gpt_oss_cfg.vocab_size, k_max=0)
    with pytest.raises(ValueError):
        Scheduler(Sim(), build_mock_pipeline(r), vocab_size=0, k_max=4)
    with pytest.raises(ValueError):  # W2: high_watermark=0 disables queue_full → unbounded queue
        Scheduler(Sim(), build_mock_pipeline(r), vocab_size=gpt_oss_cfg.vocab_size, k_max=4, high_watermark=0)


def test_call_when_idle_queues_multiple(gpt_oss_cfg):
    """M3: two idle callbacks registered while work is in flight both fire — the second
    no longer clobbers the first (a second eviction in a drain window must not be dropped)."""
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    sim = Sim()
    sched = Scheduler(sim, build_mock_pipeline(r), vocab_size=gpt_oss_cfg.vocab_size, k_max=2)
    fired = []
    sched.submit(Stream(id="s", prompt=[1, 2], max_new_tokens=5))
    sched.call_when_idle(lambda: fired.append("a"))
    sched.call_when_idle(lambda: fired.append("b"))
    sim.run()
    assert fired == ["a", "b"]


def test_multistream_improves_occupancy(gpt_oss_cfg):
    """spec §4.1/§6: single-stream wastes the pipe (≈1/N occupancy); K≥N streams fill it."""
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    n = r.n
    assert n >= 3
    vocab = gpt_oss_cfg.vocab_size

    sched_one, _ = _run(lambda: build_mock_pipeline(r), [("s", [1, 2], 40)], vocab, k_max=1)
    many = [(f"s{i}", [1, 2], 40) for i in range(n + 2)]
    sched_many, _ = _run(lambda: build_mock_pipeline(r), many, vocab, k_max=n + 2)

    occ_one = sched_one.avg_occupancy()
    occ_many = sched_many.avg_occupancy()
    assert occ_one < 0.5          # a single stream cannot fill the pipeline
    assert occ_many > occ_one     # multi-stream fills the bubble
    assert occ_many > 0.5
