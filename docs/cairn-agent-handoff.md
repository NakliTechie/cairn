# Cairn — Agent Handoff

> **Doc 3 of 3** (Vision & Roadmap → Spec → **Agent Handoff**). For Claude Code.
> Reads on top of `cairn-spec.md` (architecture, invariants, recovery model, gates)
> and `cairn-vision-roadmap.md` (thesis, regime, economics). **Do not re-derive what
> those fix — build to them.** Working name **Cairn** (trademark gate open).
> **Status:** v0.1.

---

## 0. What you're building

A commercial hosted-inference service that serves one large open MoE model by
**pipeline-splitting it across a fleet of cheap single-GPU spot instances**, kept
utilised by **multi-streaming**, and kept alive through spot interruptions by a
**reassign-first recovery loop**. The mechanism is a hard fork of
[`leyten/shard`](https://github.com/leyten/shard); the net-new is the multi-stream
scheduler and the control/fleet glue. Read the spec's §0 invariants first and keep
them open while you build — they are the contract every component answers to.

---

## 1. Doctrine context (what binds here)

This is a **commercial venture on Edge-First's commercial track**, not a NakliTechie
sovereign tool. So:
- **Sidecar does not apply** — the model's output *is* the product (AI-native).
- **A real backend is permitted** — Cloudflare control plane + AWS data plane,
  billing later. No sovereign zero-retention claim is made (see spec §8).
- **Build-vs-Adopt still binds:** adopt SkyPilot, fork Shard, build only the
  scheduler + gateway + fleet controller. Do not reimplement what the fork or
  SkyPilot already do.
- **`/forward-pass` between chunks still binds** (§9). `/walkthrough` does **not** —
  there is no UI/RBAC in v1.0; its structural cousins (replay reconstructs state,
  crypto vectors at boot) do apply and are gate artifacts.
- **Not added to the NakliTechie public portfolio or profile** (§14).

---

## 2. Repo structure

Single repo, clear seams:

```
cairn/
  fork/        # forked + stripped Shard: block runtime (SGLang wrap) + LAN transport
  scheduler/   # the multi-stream scheduler — the core build
  control/     # CF Workers gateway + Durable Objects fleet state (wrangler project)
  infra/       # SkyPilot configs, instance bootstrap, weight-staging scripts
  configs/     # model-as-config: gpt-oss-120b.yaml, qwen3.5-397b.yaml, …
  bench/        # benchmark harness + gate-artifact tests (correctness, interruption, crypto)
  docs/        # the three docs
```

`configs/` is load-bearing — **the model is configuration** (spec invariant #3). A
new model is a new YAML (layer count, per-layer + embedding + lm_head VRAM, quant,
tokenizer), never new code paths.

---

## 3. The fork — strip, keep, pin

Hard fork (Build doctrine: fork the parent and strip it, do not extract a shared
engine). **Pin to a specific upstream commit; do not plan to stay mergeable.**

**Strip out entirely:**
- NAT hole-punching, STUN/relay fallback — we are LAN-in-VPC (invariant #6).
- c0mpute integration, worker tokens, USDC/payment, referrals, permissionless join.
- The privacy / boundary-pinning roadmap code — we own every node (invariant #6).
- Per-node identity issuance / keyed-handshake scaffolding — PSK suffices (spec §8).

**Keep and re-point:**
- Contiguous-layer split + the block runtime (SGLang wrap). `forward(hidden, kv_meta)
  → hidden`, per-node KV for its block.
- Supervised edges (health, timeout, reconnect) — re-pointed at VPC LAN. **Load-
  bearing for recovery detection — do not weaken.**
- Wire format: JSON header + raw tensor bytes, **no pickle**, ChaCha20-Poly1305 under
  `SHARD_PSK`. Crypto self-test at boot, fail loud.

**Validate at build (do not assume):** the exact **SkyPilot ↔ fork ↔ controller
seam**. SkyServe's abstraction is *N identical replicas*; our topology is a *pipeline
of heterogeneous-block nodes*, which is not that. Expect to adopt SkyPilot at the
**instance provisioning + per-instance spot-recovery** layer and own block-
assignment + topology + re-stitch in our controller. Confirm which SkyPilot
primitive (managed cluster vs job vs serve) is the right one before committing.

---

## 4. Build & deploy — two planes

- **Control plane (Cloudflare):** `wrangler deploy` — Workers gateway + Durable
  Objects. Versioned; surfaced at a `/version` endpoint + meta.
- **Data plane (AWS spot):** SkyPilot launch from `infra/` — provisions the
  single-GPU spot pool in one VPC/placement group, stages weights to instance/EBS,
  boots the block runtimes + scheduler, registers nodes with the control plane.
- **Two-timescale recovery (implementation note on spec §5):** when a node dies, the
  **warm spare** takes its block in **seconds** (keeps serving); **SkyPilot**
  backfills a replacement instance in **minutes** (which becomes the *new* warm
  spare, restoring the safety margin). The warm spare covers the gap; SkyPilot
  refills the slot. Build both halves and keep them at their own timescales.

---

## 5. Environment & secrets

In a secret store (CF secrets / AWS Secrets Manager), never in the repo or an image
(spec §8). Rotatable.
- `HF_TOKEN` — block-weight pull.
- AWS credentials — SkyPilot provisioning.
- `SHARD_PSK` — wire seal.
- Cloudflare/wrangler auth — control-plane deploy.

---

## 6. Dev / iteration path — cost-laddered

Do **not** spin a full spot fleet to iterate logic. Climb the ladder:

1. **Mock-runtime (no GPU):** block runtimes are stubs that pass tensors with
   simulated per-stage latency. Exercise the scheduler's queueing, admission, K
   bounds, backpressure, and the **recovery orchestration** (drain → spare load →
   re-stitch → re-prefill → resume) with zero GPU cost. Most scheduler and
   control-plane bugs die here.
2. **Small-model real run:** split a small model (a 7–9B) across 2 cheap spot nodes
   (or 2 GPUs on one box) to prove **split correctness + real KV rebuild + the wire**
   cheaply.
3. **Full gate run:** gpt-oss-120b on the real N≈3–4 g6 pool — the v1.0 gate.

Probe and pin every dependency; never surprise-download model gigabytes without an
explicit step (Build doctrine).

---

## 7. Milestone build plan — large chunks, autonomous

Build in milestone-sized chunks, not step-by-step. Proceed autonomously on naming,
implementation choices, and debugging. Stop only on the §10 triggers. Run
`/forward-pass` at each chunk boundary and clear its list before opening the next.

**Chunk 1 — v1.0 (correctness on controlled spot, single-stream).**
Fork + strip; model-config loader; the fit algorithm (spec §3, reserving KV headroom
for K from day one); single-stream pipeline on a SkyPilot-managed g6 pool;
`RecoveryPolicy` with reassign + rebuild + the 1 warm spare; the benchmark harness.
→ produce the v1.0 gate artifacts (§8).

**Chunk 2 — v1.1 (multi-stream — the core).**
The async, queue-driven scheduler (per-stage input queues, admission control within
`[K_min=N, K_max]`, backpressure, round-robin fairness — spec §4); utilisation +
cost/token instrumentation; **the cross-stream KV-isolation correctness gate**.
→ produce the v1.1 gate artifacts.

**Chunk 3 — v1.2 (fleet hardening + spot economics).**
Multi-AZ/region spot hunt; warm-pool + on-demand floor tuning; block reassignment
under real churn at scale; eviction + thrash telemetry.
→ produce the v1.2 gate artifacts.

---

## 8. Gate artifacts — the checklist (from spec §9)

Nothing advances with a gate open. Each artifact is evidence, committed under
`bench/`.

**v1.0**
- [ ] Reliability log: N×g6.xlarge serving gpt-oss-120b, ≥ X/X clean completions.
- [ ] Correctness: split output token-for-token identical to a single-reference run
      (greedy).
- [ ] Induced-interruption test: kill a node mid-decode → reassign loads the block on
      the warm spare, re-prefills KV, re-stitches edges, resumes; request completes,
      output uncorrupted; recovery timeline logged.
- [ ] Crypto self-test passes at boot; a tampered frame resets the edge.

**v1.1**
- [ ] Per-stage GPU occupancy ≥ floor under K concurrent streams.
- [ ] cost/token beats the on-demand single-large-instance baseline, in-regime.
- [ ] **Cross-stream correctness:** token-for-token across all K streams vs single-
      stream reference. **Hard gate — this is the bug that ships silently.**

**v1.2**
- [ ] Endurance: multi-hour serving through *real* spot interruptions at target
      blended cost.
- [ ] Thrash bound: reload-overhead / total-spend < Y%.
- [ ] Eviction-rate telemetry feeding warm-pool sizing.

(X, Y, the occupancy floor, and the baseline number are set from the spec §12 calcs
at build, not guessed now.)

---

## 9. Build cadence

- **`/forward-pass`** between every chunk — lists logical + security issues; fix the
  list before the next chunk. A security issue on a core surface (secret handling,
  the wire, the recovery loop) is never deferred.
- **`/walkthrough` — N/A** (no UI/RBAC in v1.0). Its structural substitutes **do**
  apply and are gate artifacts: replaying a stream's token-history reconstructs its
  KV (spec invariant #2), and crypto vectors run at boot and fail loud.

---

## 10. Agent autonomy & escalation

**Proceed autonomously** on: naming, library/implementation choices, debugging,
file/module layout, the spec §12 calcs, tuning K and warm-pool dials. Make the call,
log the decision, keep moving.

**Stop and ask only when:**
1. **A locked decision looks wrong on contact with reality** — a spec §0 invariant,
   the recovery model (§5), or the model/instance/N rule (§3) conflicts with
   something you discover. Surface it; don't silently route around it.
2. **A new dependency is needed** beyond {Shard fork, SGLang, SkyPilot, the CF stack,
   Hugging Face}.
3. **Genuine scope ambiguity that changes the product** — not an implementation
   detail. Implementation details you decide.

When you stop: state the conflict, the options, and your recommendation. One message,
then wait.

---

## 11. What NOT to do — hard rules

Each maps to a spec invariant or a doctrine. Violating one is a defect even if it
"works."

1. **Never put Cloudflare in the per-token hot path.** The decode loop is wholly
   in-VPC; the entry node drives it. (Invariant #4.)
2. **Never use tensor-parallel or expert-parallel across nodes.** Contiguous-layer
   split only; MoE routing stays intra-stage. (Invariant #1.)
3. **Never migrate or back up KV.** Recovery is replay from the durable token-history.
   (Invariant #2 / spec §5.3.)
4. **Never add a per-model code path.** New model = new config. (Invariant #3.)
5. **Never use pickle on the wire.** JSON header + raw bytes only.
6. **Never cold-start a replacement inside the eviction window expecting a weight
   download.** Spares are pre-staged; that is the entire recovery design.
7. **Never skip the cross-stream KV-isolation gate.** (Spec §4.4.)
8. **Never build the v1.3+ deferred items now** — spec-decode, multi-model packing,
   Quiver mounting, billing/auth/multi-tenant, AZ-spread, adaptive sizing,
   activation checkpointing. Building them early is scope creep; they are deferred
   deliberately.
9. **Never optimise transport bandwidth (the activation codec) for v1.0.** It is a
   non-problem in-VPC (spec §6); leave it a knob, off.
10. **Never introduce a coordination or retention server** beyond the defined CF
    control plane, and never persist prompt/output content (spec §8).
11. **Never ship without confirming the license** — gpt-oss Apache terms (Qwen3.5
    Apache is confirmed). License is a v0 gate.

---

## 12. The agent face

This product's surface **is** machine-callable by construction: the
**OpenAI-compatible inference API** is the agent face. There is no separate
automation surface to build — one ingress, one API (invariant #7). v1.0's "human
face" is the benchmark harness + the metrics/logs; a human ops/admin console is a
later, separate concern. Do not build a second API for agents; the inference API is
already it.

---

## 13. README scope

Write `README.md` covering: what Cairn is (one paragraph, plain — frontier-size open
model served on cheap spot); the three-plane architecture in brief; how to deploy the
control plane (`wrangler`) and launch the data plane (SkyPilot); the model-as-config
system and how to add a model; how to run the benchmark harness and where the gate
artifacts live; the environment/secret variables. No marketing, no model name-drops
beyond the served target, no line counts.

---

## 14. Versioning & portfolio

- **Visible version string**, updated before every push: a `/version` endpoint on the
  gateway + a meta value.
- **Commercial — not published to the NakliTechie public portfolio or profile**, and
  not added to `NAKLITECHIE-PROJECT-STATE.md` as a public entry. It is a separate-
  company build; track it on the commercial side only.
