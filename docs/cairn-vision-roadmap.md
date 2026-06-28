# Cairn — Vision & Roadmap

> **Status:** the core mechanism is proven live. Tail hot-swap, multi-stream
> throughput, and self-replenish all run on a real spot fleet; full
> position-independent recovery is partially live and partially CPU-proven, with
> a live re-validation owed (see §7). This document states the thesis, regime,
> economics, and roadmap as built.

**One-liner.** Open-source inference that serves a single large open model by
pipeline-splitting it across a fleet of cheap, individually-reclaimable spot
GPUs, kept utilised by multi-streaming, and kept alive through preemption by a
warm-spare hot-swap recovery loop. Frontier-class models on commodity spot —
scoped to the VRAM-and-availability gap where no affordable single instance fits
the model.

---

## 1. Thesis

Big open models need more VRAM than any affordable, reliably-available single
spot instance offers. The large multi-GPU SKUs (p4d / p5, 8×A100/H100) are the
scarcest, most aggressively-reclaimed, most-fought-over pools on the spot
market. The small single-GPU pools are deeper, cheaper, and more stable.

Pipeline parallelism assembles one big model's VRAM out of many small GPUs,
shipping only the **activation tensor** at block boundaries — Ethernet-tolerant,
unlike tensor-parallel which needs NVLink-class bandwidth and dies across
separate instances.

So: assemble the model across a managed fleet of small cheap spot instances.

- **Interruption blast radius is 1/N, not total** — lose one node, replay one
  block onto a warm spare, not the whole model.
- **Rebalance onto whatever's cheapest right now** — a fixed SKU can't.
- **Multi-stream to keep every stage busy** — without it the split is nonsense.

The base mechanism is built on **pipeline-parallel block serving** (SGLang
per-block). Running it on owned spot inside one VPC deletes the three hardest
problems of decentralised pipeline inference — none of which apply once you own
every node — and adds the one thing that approach lacks: a recovery loop tuned
for spot preemption.

| Hard problem of decentralised pipeline inference | Why it vanishes here |
|---|---|
| WAN transport / NAT hole-punching | We own the network — LAN inside one VPC. |
| Volunteer-node privacy (activation leakage) | We own every node. No malicious middle node. |
| Decentralised payments / permissionless join | Single operator, controlled fleet. |
| **Recovery on preemption** | **The net-new work — warm-spare hot-swap, see §4 / §8.** |

---

## 2. Scope boundary — where this wins, and where it must not pretend to

The product has exactly one home. Outside it, a single box wins and we say so.

| Regime | Winner | Why |
|---|---|---|
| Model fits an affordable multi-GPU spot box | **One box** | NVLink/PCIe beats Ethernet; one reservation; zero pipeline code. Don't build for this. |
| Model **exceeds** the largest affordable/available single spot instance | **Cairn** | You're assembling VRAM across instances anyway — the only question is doing it well. |
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
   Multi-region appears only as a spot-supply hedge (§7), never mid-request.
4. **Not solving activation-leakage privacy.** We own every node; the problem is
   out of scope by construction.
5. **Not single-box hosting.** That is the thing we differentiate *from*.
6. **Spec-decode is a later latency lever, not a v1 pillar** (§7).

---

## 3. Architecture

```
EDGE / CLIENT          CONTROL PLANE (Cloudflare)        DATA PLANE (AWS spot)
request streams  ──►   gateway / router (Workers)   ──►  pipeline of small spot GPUs:
(K concurrent,         fleet state (Durable Objects)      node0[L0–k] → node1[k–2k] → … → nodeN
 multiplexed)    ◄──   drain / retry / reassign      ◄──  each = one cheap single-GPU spot instance
                                                          activations stream stage→stage over VPC LAN
```

- **Control plane — Cloudflare** (Workers gateway + Durable Objects for
  fleet/stream state). Cheap, global, stateful where it needs to be.
- **Data plane — AWS spot.** The GPUs live where the spot market is deep. The
  live headline run is on **g7e.2xlarge (RTX PRO 6000 Blackwell, sm_120,
  96 GiB VRAM)** in **us-east-2 (Ohio)**.
- **Per-node block serving — SGLang, wrapped per-block.** An internal detail, not
  a product surface. We do not rebuild kernels / attention / paging / quant.

---

## 4. Net-new vs adopted — the build/adopt line

The rule: adopt mature, permissively-licensed OSS; build only the shape it can't
provide.

| Layer | Decision |
|---|---|
| Spot provisioning, interruption recovery primitives, multi-region hunting, on-demand fallback | **Adopt SkyPilot / SkyServe** (Apache-2.0). |
| Model split, activation transport, per-node block serving | **Build on SGLang pipeline-parallel block serving.** Owned single-VPC LAN; no WAN / NAT / privacy / payments machinery. |
| **Warm-spare hot-swap recovery** — replay the dead block's token log onto a pre-staged spare, no NCCL re-form | **BUILD. Net-new.** |
| **Multi-stream scheduler** — multiplex K sequences through the pipeline, fill bubbles | **BUILD. Net-new.** No fleet manager does pipeline multi-streaming. |
| Gateway / router + fleet-state control plane | **Build** (CF Workers + Durable Objects). |
| Billing / auth / multi-tenant | **Out of scope.** Cairn is an open-source research artifact, not a hosted service. |

The net-new effort is spent on the two things existing tools don't provide: the
warm-spare hot-swap recovery loop and the multi-stream-over-pipeline scheduler.
Everything else is adopted.

---

## 5. Stance

Cairn is an **open-source research artifact**, not a commercial hosted service.
The deliverables are a working system, reproducible benchmarks, a paper, and a
repo people can run themselves (see `cairn-objectives.md`). There is no billing,
no multi-tenancy, no SaaS. Any monetization is **consulting** — the work lands
engagements; it is not a product to sell.

Standalone project: it adopts mature OSS where it can and builds only the
recovery loop, the scheduler, and the glue.

---

## 6. Economics — honest version

**This is a supply-arbitrage play, not an efficiency play.** A single box is more
FLOPS-efficient (no inter-node comms, no pipeline bubble). We accept the
inefficiency *only* because a single box of sufficient VRAM is unavailable or far
more expensive on spot in the target regime.

```
cost win  =  cheap small-spot prices
           − multi-stream overhead
           − warm-spare / on-demand floor
           − interruption-rebuild waste
```

- **Multi-stream is not optional polish.** Single-stream, N−1 stages sit idle —
  you pay for N GPUs and get one GPU's throughput. Multi-stream is what makes
  cost/token defensible. On the live fleet it lifts throughput from
  **6.1 → 24.9 tok/s**.
- **"Dramatic" cost reduction is real** *relative to on-demand large instances and
  managed APIs, in-regime, if interruption handling doesn't thrash.* Out of
  regime, or against an efficient single-node competitor on a model that fits
  their box, the claim does **not** hold. Pitch it the narrow way.
- **Two levers carry the system, both conditional:** the spot blend (eroded by the
  warm-spare/on-demand floor) and utilisation via multi-stream. Spec-decode and
  adapter mounting are *additional, multiplicative* levers layered later — not
  assumed in the headline.

---

## 7. Roadmap — gates inline

Gates are pass/fail. Nothing advances with a gate open. Status reflects what is
live today.

### v1.0 — Correctness on controlled spot (single-stream) · **DONE**
SGLang pipeline-parallel block serving on a SkyPilot-managed spot fleet; one
VPC/region; LAN transport; a target model that exceeds an affordable single
instance; single stream. No auth, no billing — a benchmark harness.
**Gate (met):** coherent output served across N small spot instances; survives
induced interruption via gateway-drain + block replay, request retried, no
corruption.

### v1.1 — Multi-stream (the core) · **LIVE**
Multiplex K concurrent sequences through the pipeline; fill bubbles; instrument
per-stage utilisation and cost/token.
**Status:** live multi-stream throughput **6.1 → 24.9 tok/s**; CPU-proven
correctness across streams, live re-validation owed at frontier scale.
**Gate:** per-stage GPU utilisation ≥ floor under concurrent load; cost/token
beats the on-demand single-large-instance baseline in-regime; output correctness
preserved across all K streams.

### v1.2 — Hot-swap recovery + fleet hardening · **PARTIALLY LIVE**
Warm-spare hot-swap: on a drop, replay the dead block's durable token log onto a
pre-staged spare; rebuild only that block's KV; no NCCL re-form; blast radius
1/N. Warm-spare + on-demand floor sizing; multi-AZ / region spot hunting;
self-replenish (auto-restage a fresh warm spare after a swap).
**Status:**
- **Tail hot-swap — live, measured:** MTTR **0.93 s / 3.26 s**, bit-identical
  output, zero dropped tokens.
- **Recovery is position-independent:** tail proven live; entry and middle
  CPU-proven; multi-stream recovery CPU-proven; live re-validation of the
  non-tail positions owed.
- **Self-replenish — live.**
**Gate:** sustained serving through *real* (not induced) spot interruptions at
target blended cost over a multi-hour run; thrash bounded — reload overhead < Y%
of spend.

### v1.3+ — Deferred levers (named, not specced)
Spec-decode latency lever (local or server draft) · multi-model packing on one
fleet · adapter mounting on the served base · SLA-style reliability tiers.

---

## 8. Risk concentration

**"If done reliably" carries the entire thesis.** The risk is not the inference —
it is handling interruptions without thrashing.

The core tension, stated bluntly: the per-node engine (SGLang) is **stateful and
slow to start** — weight load + CUDA-graph capture is minutes for a big model —
while spot gives only a **~2-minute interruption warning**. You cannot cold-start
a replacement inside the window. The answer is a **warm-spare hot-swap**: keep a
pre-staged spare hot, and on a drop **replay the dead block's token log** onto it
in seconds rather than migrating KV across AZs. KV is **rebuilt, not migrated** —
cheaper to redo the block than to move the state.

This is exactly the loop now proven live on the tail (MTTR 0.93 s / 3.26 s,
bit-identical, zero drops) and self-replenishing afterward. The remaining work is
the live re-validation of recovery at non-tail positions and under real
(not induced) preemption at frontier scale — the gate in §7 / v1.2.

---

## 9. Open decisions — to confirm at frontier-scale re-validation

1. **Recovery at scale** — warm-spare sizing, on-demand floor %, behaviour under
   real (not induced) multi-AZ preemption, and live re-validation of non-tail
   recovery positions.
2. **Instance family for the pool** — chosen by spot depth and $/VRAM; the live
   headline uses g7e.2xlarge (RTX PRO 6000 Blackwell) in us-east-2.
3. **Region / AZ strategy** — single placement group for the headline run;
   multi-region hedge for supply.
