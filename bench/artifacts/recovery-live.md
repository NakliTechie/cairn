# Cairn — LIVE recovery gate artifacts (real fleet)

**Model:** `Qwen/Qwen2.5-0.5B-Instruct` (24 layers, split 12/12) · **Hardware:** AWS `g6.xlarge` spot
(1× NVIDIA L4, sm_89), `eu-south-2` · **Date:** 2026-06-22

> The REAL-FLEET counterpart of the rung-1 sim "induced interruption" gate ([`sim-gates.md`](sim-gates.md)):
> live warm-spare recovery on actual sglang paged-KV. A node is hard-killed mid-decode (`CAIRN_DIE_AFTER`,
> a faithful proxy for a spot reclaim — the edge closes identically); the driver re-stitches the entry to a
> pre-warmed spare, replays the committed token history (rebuilding the spare's KV), and resumes. The gate
> is: **recovered output bit-identical to the no-death reference.**

**The recovery gate passes on real hardware: ✅ YES — bit-identical in every configuration.**

| Configuration | Output vs no-death reference | MTTR | Detail |
|---|---|---|---|
| **1 box, 3 nodes (localhost)** | ✅ identical `[5,9,3,5,9,3,5,9]` | **0.58 s** | kill tail @4 tokens → re-stitch → replay → resume; nodes share one box's kernel cache (spare effectively warm) |
| **3 separate boxes (VPC LAN)** | ✅ identical `[5,9,3,5,9,3,5,9]` | **38.9 s** | same recovery, over the real network; MTTR dominated by the spare's *cold first-forward* flashinfer JIT compile |
| **3 separate boxes + pre-warmed spare** | ✅ identical `[5,9,3,5,9,3,5,9]` | **0.023 s** | spare's flashinfer kernels compiled at load (`SglangNodeRuntime.warmup()`) — the ~38 s compile moved off the recovery path |

**Headline:** with a genuinely pre-warmed warm spare, a node dies mid-decode and the system resumes the
**identical** answer in **23 ms**, across separate machines over the VPC LAN (~1,700× faster than the
cold-spare path).

### Method
- **Induced death:** `CAIRN_DIE_AFTER=N` → the tail node `os._exit(137)` after N forwards. The edge closes,
  so the driver detects the death exactly as a real spot reclaim would present. (Real EC2 instance-kill is
  the next fidelity rung; process-death is a faithful proxy for the recovery *logic*.)
- **Recovery (Path 1, driver-orchestrated):** `serve.py set_next` re-stitch + `recovery.py
  decode_with_recovery` — replay the committed history under a *fresh seq* so the surviving entry + the
  fresh spare both prefill cleanly (rebuild KV), last logit = the pending token, resume.
- **Rigs:** `cairn_node/recovery.py prove_recovery` (1-box, localhost) ·
  `infra/skypilot/cairn-recovery.sky.yaml` (3 separate boxes over the VPC LAN).
- **Reference:** the no-death run on the same model/split/prompt. Gate = recovered tokens == reference.
- **Commits:** `f301dc3` (fresh-seq KV rebuild) · `1c98ddd` (gang-scheduler: induced death reports success)
  · `7e3ae0d` (spare pre-warm).

### Honest scope (what this proves — and does NOT)
- **Proves:** the recovery *mechanism* is correct and fast on real sglang paged-KV — single-box AND across
  separate boxes over the VPC LAN — with a pre-warmed spare bringing MTTR to ~23 ms.
- **Does NOT prove (yet):** the same at headline scale (`GLM-5.2`, multi-GPU) — this is a
  0.5B model, one L4 per box; nor is it a **cost claim** — that needs the real-fleet run + the actual AWS
  bill (§12 / Chunk C). No $ claim is made here (spec §6: "pitch it narrow").
