"""In-sim gate-artifact harness (handoff §8, spec §9).

Drives the rung-1 simulation end-to-end and emits the **sim analogs** of the v1.0 + v1.1
gate artifacts. These are NOT the real-fleet gates (those need the Shard fork + a GPU
pool and are produced by this harness's real-fleet counterpart later) — they prove the
*orchestration logic* is correct before any GPU spend, which is the entire point of the
§6 cost-ladder's rung 1.

Run:  uv run --python 3.9 --with pyyaml python bench/sim_gates.py
Emits: bench/artifacts/sim-gates.json  +  bench/artifacts/sim-gates.md
"""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scheduler" / "src"))

from cairn_scheduler import fit, load_model_config  # noqa: E402
from cairn_scheduler.recovery import RecoveryManager  # noqa: E402
from cairn_scheduler.runtime import MockBlockRuntime, build_mock_pipeline  # noqa: E402
from cairn_scheduler.scheduler import Scheduler, Stream  # noqa: E402
from cairn_scheduler.sim import Sim  # noqa: E402

CFG = ROOT / "configs" / "gpt-oss-120b.yaml"
ART = ROOT / "bench" / "artifacts"

# Illustrative prices ($/node-hour) — REAL numbers come from spec §12/§6 at build.
G6_SPOT_HR = 0.30
BASELINE_ONDEMAND_HR = 4.00  # one on-demand box large enough to hold gpt-oss-120b (~63 GB)


def _run(r, specs, vocab, *, k_max, evict_stage=None, after="s0", count=0,
         warm_spares=1, version_skew=False, high_watermark=8):
    sim = Sim()
    sched = Scheduler(sim, build_mock_pipeline(r), vocab_size=vocab, k_max=k_max,
                      high_watermark=high_watermark)
    rm = RecoveryManager(sim, sched, warm_spares=warm_spares)
    if evict_stage is not None:
        rm.arm(evict_stage, after_stream=after, after_count=count, version_skew=version_skew)
    sched.submit_all([Stream(id=i, prompt=list(p), max_new_tokens=m) for i, p, m in specs])
    sim.run()
    return sched, rm, {s.id: s.generated for s in sched.finished}


def gate_reliability(r, vocab):
    specs = [(f"s{i}", [i + 1, i + 2], 12) for i in range(20)]
    sched, _, _ = _run(r, specs, vocab, k_max=8)
    completed = len(sched.finished)
    return {"name": "v1.0 reliability", "completed": completed, "submitted": 20,
            "pass": completed == 20, "detail": f"{completed}/20 clean completions"}


def gate_correctness(r, cfg):
    vocab, L = cfg.vocab_size, cfg.num_layers
    spec = [("s", [11, 22, 33, 44], 16)]
    _, _, ref = _run_single_node(L, spec, vocab)
    _, _, split = _run(r, spec, vocab, k_max=4)
    ok = split["s"] == ref["s"]
    return {"name": "v1.0 correctness (split == single reference)", "n_stages": r.n,
            "tokens": len(split["s"]), "pass": ok and r.n >= 2,
            "detail": f"split N={r.n} token-for-token {'==' if ok else '!='} single-node reference"}


def _run_single_node(num_layers, specs, vocab):
    sim = Sim()
    sched = Scheduler(sim, [MockBlockRuntime(0, 0, num_layers - 1)], vocab_size=vocab, k_max=1)
    sched.submit_all([Stream(id=i, prompt=list(p), max_new_tokens=m) for i, p, m in specs])
    sim.run()
    return sched, None, {s.id: s.generated for s in sched.finished}


def gate_interruption(r, vocab):
    specs = [(f"s{i}", [i + 1, i + 2, i + 3], 15) for i in range(4)]
    _, _, ref = _run(r, specs, vocab, k_max=4)
    sched, rm, got = _run(r, specs, vocab, k_max=4, evict_stage=r.n // 2, after="s0", count=5)
    ev = rm.timeline[0]
    ok = got == ref and len(sched.finished) == 4
    return {"name": "v1.0 induced interruption (kill mid-decode → reassign → resume)",
            "pass": ok, "policy": ev.policy, "stage_killed": ev.stage,
            "recovery": {"t_warning": ev.t_warning, "load_s": ev.load_s,
                         "reprefill_s": round(ev.reprefill_s, 3), "t_resume": round(ev.t_resume, 3),
                         "replay_tokens": ev.replay_tokens, "affected_streams": ev.affected_streams},
            "detail": f"stage {ev.stage}/{r.n} {ev.policy}; output uncorrupted; resumed at t={ev.t_resume:.1f}"}


def gate_occupancy(r, vocab, floor=0.80):
    curve = {}
    for k in (1, 2, r.n, r.n + 4):
        sched, _, _ = _run(r, [(f"s{i}", [1, 2], 40) for i in range(k)], vocab, k_max=k)
        curve[k] = round(sched.avg_occupancy(), 3)
    at_n = curve[r.n]
    return {"name": "v1.1 utilisation (per-stage occupancy under K streams)", "n_stages": r.n,
            "floor": floor, "occupancy_vs_k": curve, "occupancy_at_k=N": at_n,
            "pass": at_n >= floor,
            "detail": f"single-stream={curve[1]} (≈1/N), K=N occupancy={at_n} (floor {floor})"}


def gate_cross_stream(r, vocab):
    specs = [(f"s{i}", [i + 1, i + 2, i + 3], 14) for i in range(6)]
    _, _, together = _run(r, specs, vocab, k_max=6)
    ok = True
    for sid, prompt, mn in specs:
        _, _, solo = _run(r, [(sid, prompt, mn)], vocab, k_max=1)
        ok = ok and together[sid] == solo[sid]
    return {"name": "v1.1 cross-stream correctness (HARD gate — the silent bug)",
            "streams": len(specs), "pass": ok,
            "detail": f"all {len(specs)} concurrent streams token-for-token == their solo runs"}


def cost_model(r, vocab):
    # Throughput from the sim under a full pipe (K = N+4), with 1 warm spare.
    k = r.n + 4
    sched, _, _ = _run(r, [(f"s{i}", [1, 2], 40) for i in range(k)], vocab, k_max=k)
    total_tokens = sum(len(s.generated) for s in sched.finished)
    sim_seconds = sched.sim.now  # one stage-forward == 1 "second" in the model
    nodes_billed = r.n + 1  # pipeline + 1 warm spare
    cairn_per_tok = (nodes_billed * G6_SPOT_HR / 3600.0) * sim_seconds / total_tokens
    # Baseline: one on-demand box doing the whole model per token; ~N stage-forwards/token.
    baseline_tokens_per_sec = total_tokens / sim_seconds  # same per-token work, no spot blend
    baseline_per_tok = (BASELINE_ONDEMAND_HR / 3600.0) / baseline_tokens_per_sec
    ratio = baseline_per_tok / cairn_per_tok if cairn_per_tok else 0.0
    return {"name": "v1.1 cost/token (ILLUSTRATIVE — real $/hr + throughput from §12)",
            "illustrative": True, "pass": None,
            "cairn_$per_Mtok": round(cairn_per_tok * 1e6, 4),
            "baseline_$per_Mtok": round(baseline_per_tok * 1e6, 4),
            "ratio_baseline_over_cairn": round(ratio, 2),
            "detail": "occupancy-driven; numbers illustrative, not a $ claim (spec §6 'pitch it narrow')"}


def run_all() -> dict:
    cfg = load_model_config(CFG)
    r = fit(cfg, target_k=8, context_len=4096)
    vocab = cfg.vocab_size
    gates = [
        gate_reliability(r, vocab),
        gate_correctness(r, cfg),
        gate_interruption(r, vocab),
        gate_occupancy(r, vocab),
        gate_cross_stream(r, vocab),
        cost_model(r, vocab),
    ]
    hard = [g for g in gates if g.get("pass") is not None]
    return {
        "model": cfg.name,
        "fit": {"n_stages": r.n, "target_k": r.target_k, "context_len": r.max_context,
                "stage_layers": [a.num_layers for a in r.assignments]},
        "gates": gates,
        "all_hard_gates_pass": all(g["pass"] for g in hard),
        "note": "Rung-1 SIMULATION gates (no GPU). Real-fleet gates need the Shard fork + a g6 pool.",
    }


def _to_markdown(result: dict) -> str:
    lines = [
        "# Cairn — rung-1 simulation gate artifacts",
        "",
        f"**Model:** `{result['model']}`  ·  **Fit:** N={result['fit']['n_stages']} stages, "
        f"layers/stage={result['fit']['stage_layers']}, K_target={result['fit']['target_k']}",
        "",
        f"> {result['note']}",
        "",
        f"**All hard sim-gates pass: {'✅ YES' if result['all_hard_gates_pass'] else '❌ NO'}**",
        "",
        "| Gate | Result | Detail |",
        "|------|--------|--------|",
    ]
    for g in result["gates"]:
        status = "ℹ️ illustrative" if g.get("pass") is None else ("✅ pass" if g["pass"] else "❌ FAIL")
        lines.append(f"| {g['name']} | {status} | {g['detail']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    result = run_all()
    ART.mkdir(parents=True, exist_ok=True)
    (ART / "sim-gates.json").write_text(json.dumps(result, indent=2) + "\n")
    (ART / "sim-gates.md").write_text(_to_markdown(result))
    print(_to_markdown(result))
    print(f"Artifacts → {ART}/sim-gates.json + sim-gates.md")
    return 0 if result["all_hard_gates_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
