# Cairn — what we've achieved

_Snapshot: 2026-06-28. A plain-English, evidence-backed account of where the project stands._

---

## The one-line version

**We serve a 600B-parameter frontier model (DeepSeek-V4-Flash) split across a fleet of
cheap, individually-reclaimable single-GPU spot instances — and when a spot box gets
reclaimed mid-generation, a warm spare takes over the dead slice and the token stream
keeps going, bit-identical, without dropping the request.** The pool then re-provisions
itself back to full strength automatically.

That combination — frontier-class model + commodity spot economics + live failover that
makes a reclaim invisible to the caller — is the thing the whole project was built to prove.
It is now proven on real hardware, end to end.

---

## What that means concretely

A model this size does not fit on any single affordable GPU. The usual answer is an
expensive multi-GPU on-demand box. Cairn's answer:

1. **Split the model by layers** across N separate single-GPU spot instances, each holding
   one contiguous block of layers. The decode loop passes the half-finished thought from box
   to box over the in-VPC LAN (encrypted wire, no pickle).
2. **Keep the fleet busy** by running many request streams concurrently through the same
   pipeline (multi-stream occupancy).
3. **Survive reclaims** with a warm spare and a token-history-is-truth replay: when a stage
   dies, its slice is re-stitched onto the spare and the committed token history is replayed
   under a fresh sequence — so generation resumes exactly where it was, no corruption.
4. **Heal the pool** — once a spare is consumed, a replacement spot box is launched
   automatically to restore the warm pool.

The honest engineering contribution is **the failover/recovery layer**, not the model. The
failover is what unlocks the 3–4× spot discount safely — it doesn't add cost, it removes the
reason you couldn't use spot for serving in the first place.

---

## Live-verified results (on real Blackwell GPUs, claim-worthy)

These are measured on a live fleet, not simulated or computed:

| Capability | Evidence |
|---|---|
| **Distributed frontier-model decode** | DeepSeek-V4-Flash FP8 (600B, 43 layers) split 4 ways across 4× g7e spot in Ohio; coherent, correct answers ("A cairn is a human-made pile of stones…", "17×24 = 408", a full ~990-word essay on Rome). |
| **Hot-swap recovery — single stream** | Tail stage drained mid-generation → warm spare re-stitched → resumed bit-identical, **zero dropped tokens**. Measured MTTR: **0.023s** (pre-warmed L4), **0.93s** and **3.26s** (V4 on Blackwell, mid-essay). |
| **Recovery is position-aware** | Proactive drain (the ~2-min spot pre-warning) carries the dying stage's identity → graceful migration before death. Reactive (abrupt SIGKILL) half-open-timeout fallback proven on the tail. |
| **Multi-stream throughput** | 6 concurrent streams served live; throughput scales near-linearly to the fleet ceiling: **6.1 → 12.4 → 24.5 → 24.9 tok/s** at 1/2/4/8 streams (≈4× occupancy gain before saturation). |
| **Self-replenishing pool** | Consuming the spare auto-fired a real `sky launch` of a replacement spot box (host-side replenish watcher). Verified live. |
| **Warm re-slicing** | A 3-way fit that OOM'd by a hair was re-cut to 4-way **on the same running boxes** from on-disk weights (~8 min) instead of a cold relaunch (~40 min). |
| **Safe spot operations** | Scoped least-privilege IAM (no admin keys), `cairn=true`-tagged teardown verified via the EC2 API as ground truth, kill-switch by tag. |

## Proven on CPU / in simulation (not yet live-confirmed)

Mechanically the same code as the live paths, validated bit-identical off-GPU:

- **Any-position recovery** (entry / middle / tail), 7/7 bit-identical.
- **Multi-stream recovery** (K streams swap together onto one spare, all replayed), 3/3.
- **Load-on-promotion reshape** (a generic spare re-execs into the dead rank's slice).
- **Catastrophic / control-plane recovery design** (durable token history mirrored off-box;
  star topology, not peer mesh) — endpoints exist and are sim-tested.

## Today's session (2026-06-28) in one glance

- First **fully coherent long-form** V4 output across the fleet (the Rome essay).
- **Live mid-essay tail swap**: drained at token 43, resumed at 44, MTTR 3.26s, essay finished
  cleanly with no visible seam.
- **Self-replenish fired live** and launched a replacement box.
- **Throughput curve characterized** (the 6.1→24.9 tok/s scaling above).
- Multi-stream live swap + entry/mid drains: **in progress / parked for next session.**

---

## The road it took to get here (condensed)

The project climbed a deliberate cost-ladder — mock-first logic, then small models, then the
headline model — and cleared a long series of real walls, most of them on workstation
Blackwell (sm_120), which had **no stock-upstream working path** for this model family:

- **Architecture & correctness first** (mock → real numerics): contiguous-layer split ==
  single-model greedy decode, token-for-token; replay-rebuild resumes bit-identical. ~110
  automated tests green before any GPU spend.
- **Deploy path**: SkyPilot → AWS spot → GPU, end-to-end, auto-torn-down.
- **The sm_120 saga**: gpt-oss-120b hit an upstream Triton-PTX ceiling; research showed the
  *same* gap blocks the whole DSA/MoE-FP4 family on workstation Blackwell (DeepGEMM refuses
  sm_120). Pivoted to a community sglang fork (0xSero) and climbed ~11 numbered "walls"
  (load path, per-layer norm under layer-split, V4 4D wire shape, SWA translation cache, the
  MoE GEMM config init order…) to reach a clean FP8 GEMM and a loading model.
- **Wall #11** turned out to be the *wrong checkpoint* (int8 has no sm_120 MoE kernel) — the
  FP8 repack was the fix. Cracked byte-level for $0 of GPU time with a shape probe + HF
  header reads.
- **The keystone** — 3-vs-4-way VRAM verdict — settled live: 3-way OOMs at ~94/95 GiB,
  4-way fits at ~74 GiB/box, re-sliced warm.

Full blow-by-blow: `plan/history.md` (decisions + per-day log + dead ends) and the
session-by-session `plan/learning-log.md`.

---

## What's still open (honest)

- **Live multi-stream recovery** and **entry/mid live drains** — CPU-proven, live runs pending
  (each needs a warm spare, ~30 min to warm a fresh box).
- **Clean multi-death** (≥2 simultaneous losses) — needs ≥2 warm spares; parked.
- **Interior reactive death** (abrupt kill of a middle box) needs heartbeats (the Path-2
  control plane) to identify *which* interior node died; proactive drain covers it today.
- **Catastrophic recovery + control plane** (Cloudflare DO durable history, resumable driver)
  — designed and sim-tested, not deployed.

See "How far from productising" below and `report/productization.md`.
