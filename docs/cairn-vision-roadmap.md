# Cairn — Vision & Roadmap

> **Working name Cairn** (trademark / domain check is a v0 gate).
> **Status:** first draft · v0.1 · pre-spec.

**One-liner.** Hosted inference that serves a single large open model by
pipeline-splitting it across a fleet of cheap, individually-reclaimable spot
GPUs, kept utilised by multi-streaming. Frontier-class models on commodity spot —
scoped to the VRAM-and-availability gap where no affordable single instance fits
the model.

---

## 1. Thesis

Big open models (100B+) need more VRAM than any affordable, reliably-available
single spot instance offers. The large multi-GPU SKUs (p4d / p5, 8×A100/H100) are
the scarcest, most aggressively-reclaimed, most-fought-over pools on the spot
market. The small single-GPU pools (g5 / g6 — A10G / L4) are deep, cheap, and
stable.

Pipeline parallelism assembles one big model's VRAM out of many small GPUs,
shipping only the **activation tensor** at block boundaries — Ethernet-tolerant,
unlike tensor-parallel which needs NVLink-class bandwidth and dies across separate
instances.

So: assemble the model across a managed fleet of small cheap spot instances.

- **Interruption blast radius is 1/N, not total** — lose one node, reassign one
  block, not the whole model.
- **Rebalance onto whatever's cheapest right now** — a fixed SKU can't.
- **Multi-stream to keep every stage busy** — without it the split is nonsense.

The mechanism already exists: **Shard** (Apache-2.0, pipeline-parallel inference,
the c0mpute engine). We **fork it and delete its three hardest problems** — none of
which apply once we own every node in our own VPC — and add the one thing it lacks.

| Shard's hard problem | Why it vanishes for us |
|---|---|
| WAN transport / NAT hole-punching | We own the network — LAN inside one VPC. |
| Volunteer-node privacy (35–59% activation leakage) | We own every node. No malicious middle node. |
| Decentralised payments / permissionless join | Single operator, controlled fleet. |
| **Multi-stream scheduling (Shard lacks it)** | **This is the net-new work — see §4.** |

---

## 2. Scope boundary — where this wins, and where it must not pretend to

The product has exactly one home. Outside it, a single box wins and we say so.

| Regime | Winner | Why |
|---|---|---|
| Model fits an affordable multi-GPU spot box (e.g. 70B on 4×A10G / 96 GB) | **One box** | NVLink/PCIe beats Ethernet; one reservation; zero pipeline code. Don't build for this. |
| Model **exceeds** the largest affordable/available single spot instance (100B+) | **Cairn** | You're assembling VRAM across instances anyway — the only question is doing it well. |
| Big multi-GPU SKUs too scarce/volatile to rely on for spot | **Cairn** | Small pools are deep and stable; blast radius = 1/N. |
| Cost-arbitrage across fragmented cheap supply (AZs / types) | **Cairn** | Fleet rebalances onto cheapest supply; a fixed SKU can't. |

**Non-goals (state these plainly, they protect the product):**

1. **Not competing on cost/token for models that fit an affordable single
   instance.** We *lose* there by construction — inter-stage overhead and
   imperfect bubble-fill make a split less FLOPS-efficient than one box. The wedge
   is the VRAM-gap + spot-availability-gap regime, nothing else.
2. **Not a decentralised / volunteer network.** Single operator, owned fleet,
   controlled VPC.
3. **Not WAN / open-internet.** LAN inside one region or placement group.
   Multi-region appears only as a spot-supply hedge (§7, v1.2), never mid-request.
4. **Not solving activation-leakage privacy.** We own every node; the problem is
   out of scope by construction.
5. **Not vLLM-hosting-on-one-box.** That is the thing we differentiate *from*.
6. **Spec-decode is a later latency lever, not a v1 pillar** (§7, v1.3+).

---

## 3. Architecture

```
EDGE / CLIENT          CONTROL PLANE (Cloudflare)        DATA PLANE (AWS spot)
request streams  ──►   gateway / router (Workers)   ──►  pipeline of small spot GPUs:
(K concurrent,         fleet state (Durable Objects)      node0[L0–k] → node1[k–2k] → … → nodeN
 multiplexed)    ◄──   drain / retry / reassign      ◄──  each = one cheap single-GPU spot instance
                                                          activations stream stage→stage over VPC LAN
```

- **Control plane — Cloudflare** (Workers gateway + DO for fleet/stream state).
  Keeps the existing CF muscle; cheap, global, stateful-where-needed.
- **Data plane — AWS spot.** The GPUs live where the spot market is deep. This is
  the portfolio's **first AWS-native data plane** — a deliberate break from the
  all-Cloudflare pattern, because CF has no spot GPU market.
- **Per-node block serving — inherited from the Shard fork (SGLang-based).** An
  internal detail, not a product surface. We do not rebuild kernels / attention /
  paging / quant.

---

## 4. Net-new vs adopted — the build/adopt line

Build-vs-Adopt doctrine: adopt mature permissively-licensed OSS; build only the
shape it can't provide.

| Layer | Decision |
|---|---|
| Spot provisioning, interruption recovery, multi-region hunting, on-demand fallback | **Adopt SkyPilot / SkyServe** (Apache-2.0). Confirm current feature set at build — versions move. |
| Model split, activation transport, per-node block serving, block reassignment | **Fork Shard.** Strip WAN / NAT / privacy / payments. |
| **Multi-stream scheduler** — multiplex K sequences through the pipeline, fill bubbles | **BUILD. The core net-new.** No fleet manager does pipeline multi-streaming. |
| Gateway / router + fleet-state control plane | **Build** (CF Workers + DO). |
| Billing / auth / multi-tenant | **Build later** (commercial track allows a real backend). Deferred past v1.0 — the P0 pattern. |

> The "new shape the existing tool can't provide" exception is spent **only** on
> the multi-stream-over-pipeline scheduler. Everything else is adopt or fork.

---

## 5. Doctrine placement

| Doctrine | Placement |
|---|---|
| **Sidecar** | **Exempt.** AI-native — the model's output *is* the product. Removability does not apply. |
| **Edge-First** | **Commercial track.** This is a C2-class relay, but commercial: full backend, billing permitted. A local-draft client is a v1.x option, not core. |
| **Build** | **Rung-3** (full client–server commercial), deliberately. First AWS-native data plane; control plane stays Cloudflare. |
| **Retention** | Commercial → real backend permitted. (A sovereign fork, if ever made, would inherit the zero-retention relay boundary. Out of scope here.) |
| **Build-vs-Adopt** | Honoured — adopt SkyPilot, fork Shard, build only the scheduler + gateway + glue. |

Standalone commercial product. Learns from the portfolio; depends on none of it.

---

## 6. Economics — honest version

**This is a supply-arbitrage play, not an efficiency play.** A single box is more
FLOPS-efficient (no inter-node comms, no pipeline bubble). We accept the
inefficiency *only* because a single box of sufficient VRAM is unavailable or far
more expensive on spot in the target regime.

```
cost win  =  cheap small-spot prices
           − multi-stream overhead
           − warm-pool / on-demand floor
           − interruption-rebuild waste
```

- **Multi-stream is not optional polish.** Single-stream, N−1 stages sit idle —
  you pay for N GPUs and get one GPU's throughput. Multi-stream is what makes
  cost/token defensible. Hence it lands in v1.1, immediately after the v1.0
  correctness proof.
- **"Dramatic" cost reduction is real** *relative to on-demand large instances and
  managed APIs, in-regime, if interruption handling doesn't thrash.* Out of regime,
  or against an efficient single-node competitor on a model that fits their box,
  the claim does **not** hold. Pitch it the narrow way.
- **Two levers carry v1, both conditional:** the spot blend (≈2–4×, eroded by the
  floor) and utilisation via multi-stream. Spec-decode and Quiver adapter mounting
  are *additional, multiplicative* levers layered later — not assumed in v1.

---

## 7. Roadmap — v1.0 → v1.x, gates inline

Gates are pass/fail. Nothing advances with a gate open.

### v1.0 — Correctness on controlled spot (single-stream)
Fork Shard onto a SkyPilot-managed spot fleet; one VPC/region; LAN transport; one
target model that exceeds an affordable single instance; single stream. No auth,
no billing — a benchmark harness. (Shard-Phase-0-on-spot.)
**Gate:** coherent output served across N small spot instances; survives ≥1
*induced* interruption via gateway-drain + block reassignment, request retried, no
corruption; reliability ≥ X/X clean completions.

### v1.1 — Multi-stream (the core)
Multiplex K concurrent sequences through the pipeline; fill bubbles; instrument
per-stage utilisation and cost/token.
**Gate:** per-stage GPU utilisation ≥ floor under concurrent load; cost/token beats
the on-demand single-large-instance baseline in-regime; output correctness
preserved across all K streams.

### v1.2 — Fleet hardening + spot economics
Warm-pool + on-demand floor sizing; multi-AZ / region spot hunting; block
reassignment under real churn at scale; cost + interruption telemetry; the
gateway's drain/retry/reassign loop hardened.
**Gate:** sustained serving through *real* (not induced) spot interruptions at
target blended cost over a multi-hour run; thrash bounded — reload overhead < Y% of
spend.

### v1.3+ — Deferred levers (named, not specced)
Spec-decode latency lever (local or server draft) · multi-model packing on one
fleet · Quiver adapter mounting on the served base · billing / auth / multi-tenant ·
SLA tiers.

---

## 8. Risk concentration

**"If done reliably" carries the entire thesis.** The risk is not the inference —
it is the fleet manager handling interruptions without thrashing.

The core tension, stated bluntly: the per-node engine (SGLang) is **stateful and
slow to start** — weight load + CUDA-graph capture is minutes for a big model —
while spot gives only a **~2-minute interruption warning**. You cannot cold-start a
replacement inside the window. The forced consequence is a **warm-pool +
on-demand floor**, and that floor erodes the saving. KV-cache should be **rebuilt,
not migrated**, on a drop — cheaper to redo the request than to move the state
across AZs.

If this loop thrashes, the savings evaporate into reload overhead and the product
has no reason to exist. **This is where Cairn lives or dies, and it is the
immediate next work.**

---

## 9. Open decisions — deferred to spec

1. **Spot-interruption architecture** — warm-pool sizing, on-demand floor %,
   reassign-block vs rebuild-pipeline policy, gateway drain/retry semantics,
   KV-cache-on-drop (rebuild, to confirm). **← NEXT conversation.**
2. **Target model for v1.0** — the one that exceeds an affordable single spot
   instance (candidate: a 100B+ open model). Pick at spec.
3. **Instance family for the pool** — g5 / g6 / A10G / L4, chosen by spot depth and
   $/VRAM. Pick at spec.
4. **Region / AZ strategy** — single placement group for v1.0; multi-region hedge
   in v1.2.
5. **Name + trademark/domain check** — v0 gate.
