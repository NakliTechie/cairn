# Cairn: A Minimal Reassign-First Failover Layer for Pipeline-Parallel LLM Serving on Reclaimable Spot GPUs

> **Status:** working draft / artifact skeleton (2026-06-28). This is the paper
> scaffold for the open-source research artifact. Every section is drafted, but
> claims are explicitly tagged by evidence tier:
>
> - **[LIVE]** — measured on real hardware (Blackwell g7e / L4 spot fleets), reproducible from `bench/` artifacts.
> - **[CPU-PROVEN]** — mechanically the same code, validated bit-identical off-GPU (unit/sim suite), not yet confirmed on a live fleet.
> - **[DESIGNED]** — specified and (in some cases) endpoint-stubbed/sim-tested, but not built or deployed end-to-end.
> - **TODO** — a number or result we intend to report but have not yet measured.
>
> A companion detailed comparison may live in `report/benchmark-plan.md` (authored
> separately); §6 references it for the full head-to-head methodology.

---

## Abstract

Frontier-class open language models (hundreds of billions of parameters) no longer
fit on any single affordable, reliably-available GPU. The conventional remedy — a
large multi-GPU on-demand instance — is exactly the scarcest, most expensive, most
aggressively-reclaimed pool on the spot market, while the deep, cheap, stable
supply lives in single-GPU instances (3–4× cheaper than on-demand). Assembling a
frontier model across many such small instances via pipeline parallelism is
well-understood; the open problem is *keeping it serving through preemption*, because
the dominant cross-node parallelism primitive (NCCL collective pipeline-parallelism)
hangs when any participant vanishes, and weight reload dwarfs the ~2-minute spot
warning.

We present **Cairn**, a systems/operations failover layer that makes a
layer-split frontier model survive individual spot reclaims with on-demand-grade
continuity at spot cost. Cairn rests on three co-designed mechanisms: (1) a strict
**contiguous-layer split** with a per-block transport that wraps SGLang and replaces
cross-node NCCL entirely, bounding the blast radius of any single loss to **1/N**;
(2) a **token-history-is-truth recovery model** in which KV-cache is treated as
derived state and recovery is replay — a dead stage's slice is re-stitched onto a
**pre-staged warm spare** and only that block's KV is rebuilt by replaying the
committed token history under a fresh sequence, resuming bit-identical; and (3) a
**multi-stream scheduler** that fills pipeline bubbles to make the split's
cost/token defensible.

We report preliminary results from a live deployment serving DeepSeek-V4-Flash
(FP8, ~600 GB on disk, 43 layers) split four ways across single-GPU Blackwell spot
instances. Single-stream hot-swap recovery completes in **0.93 s and 3.26 s**
(mid-generation, bit-identical, zero dropped tokens) **[LIVE]**; multi-stream
throughput scales **6.1 → 24.9 tok/s** from one to eight concurrent streams
**[LIVE]**; and the warm pool self-replenishes a consumed spare automatically
**[LIVE]**. The remaining recovery matrix (multi-stream live swap, entry/mid drains,
multi-death) is bit-identical in simulation and pending live confirmation
**[CPU-PROVEN]**. We are explicit that the contribution is the failover layer, not
the model or any kernel; that the standing warm-spare and its ~20–40 min warm-up
latency are the principal operational ceiling; and that the durable control plane is
designed but not yet deployed.

---

## 1. Introduction

### 1.1 The gap

Three facts collide.

**Frontier-class open models do not fit any affordable single GPU.** A 100B–750B
model, even quantized, needs hundreds of gigabytes of VRAM — far beyond a single
commodity card. The model is forced across multiple GPUs no matter what.

**The instances that *can* hold such a model are the worst place to rent on spot.**
Large multi-GPU SKUs (8×H100/A100-class) are the scarcest, most contended, most
aggressively-reclaimed pools on the spot market. The deep, cheap, stable supply is
in *single-GPU* instances — and spot capacity there is roughly **3–4× cheaper than
on-demand**.

**Spot is individually reclaimable.** A spot instance can be taken back with a short
(~2-minute) warning, or hard-killed. The naive way to assemble a model across nodes —
NCCL collective pipeline/tensor parallelism — turns a single reclaim into a
whole-pipeline hang: collectives block on the departed rank with no clean failure.
And a replacement cannot be cold-started inside the warning window, because loading
weights and capturing CUDA graphs for a frontier model takes minutes, not seconds.

So the affordable supply is exactly the supply you cannot naively use for a model big
enough to need it.

### 1.2 The thesis

Cairn's thesis is that an **owned-fleet, contiguous-layer split** plus a
**reassign-first failover layer** unlocks spot economics at on-demand-grade
reliability — precisely in the regime where the split is *mandatory* (the model
exceeds any single affordable instance) rather than merely economical.

The key move is to stop treating the running system's state as precious. **Token
history is the source of truth; the KV-cache is derived.** A dead stage is not
recovered by migrating its KV (there is no copy, and continuous KV replication would
swamp the wire); it is recovered by **replay** — re-prefilling only the dead block on
a pre-staged warm spare over the committed token history, then resuming. Because the
blocks are *fixed* (no reparallelization), the blast radius of any loss is **1/N**,
and recovery touches exactly one block.

This is deliberately the *minimal* recovery primitive. It does not reparallelize the
model (SpotServe) and it does not continuously replicate KV (KevlarFlow). The bet —
and the thing this artifact exists to test — is that the minimal primitive stays
competitive with those richer mechanisms at frontier scale, at a fraction of the
complexity.

### 1.3 Contributions

1. A **failover layer for layer-split LLM serving on preemptible spot** that wraps
   SGLang per-block and replaces cross-node NCCL with its own encrypted wire,
   transport, and recovery — making a single reclaim a 1/N event rather than a
   pipeline-wide hang.
2. A **token-history-is-truth recovery model**: replay-rebuild of only the dead
   block's KV onto a pre-staged warm spare, with a **proactive drain** (on the spot
   pre-warning) as the primary path and a **reactive half-open-timeout** fallback,
   resuming bit-identical under a fresh sequence.
3. A **multi-stream scheduler** that fills pipeline bubbles across the split (the
   net-new component — no off-the-shelf fleet manager does pipeline multi-streaming).
4. A **self-replenishing warm pool** and a **load-on-promotion reshape** that lets a
   generic spare re-exec into any dead rank's slice, plus a designed
   **durable-STAR catastrophic-recovery** path (token history mirrored to a
   Cloudflare Durable Object; a resumable driver).
5. **Preliminary live evidence** on a frontier model (DeepSeek-V4-Flash FP8) and an
   honest map of what is proven live, proven in simulation, and designed-but-unbuilt.

### 1.4 Honest scope (stated up front, not buried)

The contribution is a **systems/ops reliability layer on top of existing serving**
(SGLang). It is **not** a new model, attention mechanism, or quantization scheme. The
failover *unlocks* the 3–4× spot discount; it does not add cost beyond one standing
warm spare. We claim "≈ on-demand reliability at spot cost," not "free." We make no
zero-retention/sovereign claim. §7 enumerates the limitations in full.

---

## 2. Background & Motivation

### 2.1 The VRAM-and-availability gap

The product has exactly one regime. Where a model fits an affordable multi-GPU box
(e.g. a 70B model on 4×A10G/96 GB), a single box wins by construction — NVLink/PCIe
beats Ethernet, one reservation, no pipeline code — and we say so. Where a model
**exceeds the largest affordable/available single instance** (100B+), VRAM must be
assembled across instances regardless; the only question is doing it well, and that
is Cairn's home. The wedge is the intersection of the **VRAM gap** (no single
affordable GPU holds the model) and the **availability gap** (the big SKUs that could
are too scarce/volatile to rely on for spot).

Pipeline parallelism is the right tool for this gap because it ships only the
**activation tensor** at block boundaries — a KB-scale hidden-state vector per token —
which is Ethernet-tolerant. Tensor-parallel, by contrast, needs NVLink-class
bandwidth and dies across separate instances.

### 2.2 The spot reclaim model

Spot capacity is rented at a steep discount in exchange for preemptibility. Two
reclaim modes matter:

- **Warned reclaim (~2 min).** The instance metadata service signals an impending
  reclaim. This is the *common, exploitable* case: the node is still alive and can
  act on the warning.
- **Hard kill (no warning).** The instance simply vanishes. Detection falls to edge
  supervision (a transport timeout), and any token in mid-traversal at the dead stage
  is lost and must be re-driven.

Two further properties shape the design. First, **a frontier model cannot be
cold-started inside the warning window** — weight load + CUDA-graph capture is minutes.
This forces a **pre-staged warm spare** (the single largest design consequence).
Second, **reclaims are often correlated** — one AWS capacity event can take many
same-type/same-AZ instances at once, denting the 1/N blast-radius assumption and able
to overrun a one-spare pool. This is why warm-pool sizing is keyed to *observed*
eviction rate, and why decorrelation (AZ/type spread) and an on-demand floor are the
hedges.

### 2.3 Why NCCL-PP fails under preemption

NCCL collectives assume a fixed, complete set of participants. When a rank
disappears mid-collective, the others block indefinitely — there is no clean,
fast-failing error; the whole pipeline hangs. Recovery requires tearing down and
re-forming the communicator, which in practice means restarting serving. This is
fundamentally incompatible with a model assembled from individually-reclaimable
nodes, where *more nodes means more reclaims*.

Cairn's response is to **not use cross-node NCCL at all**. It wraps SGLang at the
*per-block* granularity (NCCL stays *inside* a node if a block ever spans local GPUs,
never across the network) and carries activations stage-to-stage over its own
supervised, encrypted wire. A dead stage is then a localized, fast-detected,
1/N event — a transport edge that times out — rather than a global collective stall.

---

## 3. Design

Cairn is three planes. The per-token hot path is wholly inside the data plane.

```
PLANE 1 — CLIENT            PLANE 2 — CONTROL (Cloudflare)        PLANE 3 — DATA (AWS spot, one VPC)
 API request          ──►   Gateway (Workers): auth, admit,   ──► Entry node ─► … ─► Tail node ─► sample
 (OpenAI-compatible)        validate, set up route                (embed+L0..)   (..lm_head)        │
 token stream         ◄──   Fleet state (Durable Objects):    ◄──        ▲───────── per-token ──────┘
                            registry · topology · health ·            (sampled token re-enters entry;
                            durable stream token-history             whole loop stays on the VPC LAN)
                            ── drain / retry / reassign ──►
```

**Invariant (control plane is out of the hot path).** The decode loop runs entirely
inside the AWS VPC; the entry node drives generation. Cloudflare does admission,
orchestration, durable stream state, and (later) billing — never a per-token round
trip. A CF↔AWS hop per token would be fatal to latency.

### 3.1 Contiguous-layer split + fit

The model is cut into **contiguous blocks of layers, one block per node**. There is
**no tensor-parallel and no expert-parallel across nodes**. For MoE models, every
layer's experts *and* its router live wholly on one stage — routing is intra-stage,
never a cross-node dispatch. This is the rule that keeps MoE tractable here: a stage
is a self-contained `forward(hidden_states, kv_meta) → hidden_states`.

The embedding is pinned to stage 0 and `lm_head` to stage N−1. The number of stages N
is **derived, not chosen**: the fit algorithm reserves, for every node,

```
block weights  +  KV headroom (K streams × max-context × this block's layers)
               +  activation buffers   ≤   node VRAM
```

KV headroom for the target concurrency K is reserved **from day one** — it is what
makes usable VRAM less than card VRAM, and it couples back to N (more KV headroom →
fewer layers per node → more stages). The algorithm is greedy-then-balanced: walk
layers, accumulate onto the current node until the next would exceed budget, open the
next node, then rebalance to **minimize the max-loaded stage** (the pipeline runs at
the speed of its slowest stage). Heterogeneous VRAM yields uneven blocks; v1.0 uses a
homogeneous pool but the algorithm supports mixed.

**Model-as-configuration.** The fit takes *any* contiguous-layer model (dense or MoE)
as input; a new model is a YAML (layer count, per-layer VRAM, embedding/`lm_head`
sizes, quant), never a code path. This is a hard invariant — there are no
model-specific branches in the scheduler. **[LIVE]** the fit ran live to settle the
DeepSeek-V4-Flash verdict: a 3-way split OOMs at ~94–95 GiB/box; a 4-way split fits
at ~74 GiB/box.

### 3.2 The encrypted per-block wire

The inter-stage transport is inherited from the Shard fork and stripped to LAN.
Because the fleet lives in one VPC/placement group, there is no NAT between stages —
**hole-punching and relay fallback are deleted entirely** — and inter-stage RTT is
sub-millisecond. The per-token activation is a single hidden-state vector (KB-scale),
far under the instances' LAN bandwidth (g6 ≤10 Gbps, g6e ≤20 Gbps), so transport
bandwidth is not the bottleneck; per-stage compute and the recovery loop are.

Two properties are load-bearing and kept:

- **Wire format:** a JSON header + raw tensor bytes, **no pickle** (a hostile frame is
  a parse error, never code execution), sealed with **ChaCha20-Poly1305** under a
  shared `SHARD_PSK`. Crypto self-test vectors run at boot and fail loud on mismatch.
  In a trusted owned VPC this is cheap defence-in-depth, and it is already built.
  **[LIVE]** crypto self-test passes at boot; a tampered frame resets the edge.
- **Supervised edges:** per-edge health, timeouts, fast fault detection, reconnect.
  This is how a dead stage is *detected* and recovery triggered — there is no
  black-box "broken pipe"; every edge logs its own health.

A per-activation codec (fp8/int8 quant of the activation tensor) is a config knob,
**off by default** — unnecessary at LAN bandwidth, retained only for the AZ-spread
case where hops cross the network.

### 3.3 Multi-stream scheduler + occupancy

A pipeline of N stages serves one token per full traversal (stage 0 → … → N−1 →
sample). Autoregressive decode is one token at a time, and token *t+1* needs *t*
sampled first, so **a single stream cannot be pipelined** — at any instant one stage
works and N−1 idle. Single-stream utilization is ≈1/N: you pay for N GPUs and get one
GPU's throughput. Multi-stream is therefore **not optional polish**; it is what makes
cost/token defensible.

The mechanism is distributed continuous batching. Run K **independent** streams
(different requests); while stream A's token is at stage 2, B's can be at stage 1, C's
at stage 0. With **K ≥ N**, every stage is busy on a different stream's token every
cycle. This is vLLM-style continuous batching spread *across* a pipeline split rather
than within one node — and nothing off-the-shelf does it across the split, which is
why it is the net-new build.

Locked design decisions:

1. **Async, queue-driven stages — not lockstep.** Each stage has an input queue of
   `(stream_id, activation)`, processes FIFO, and emits to the next stage's queue.
   The pipeline self-fills as long as ≥N streams are active. Lockstep would stall the
   whole pipeline on the slowest stage every cycle.
2. **K bounds.** `K_min = N` (saturate); `K_max` is bounded by per-stage KV VRAM
   (every stage holds KV for all K streams through its block). The central tension:
   K must be ≥ N to fill the pipe and ≤ the VRAM ceiling.
3. **Admission control.** Admit up to `K_max`; never below `K_min` while work exists.
   A new stream runs prefill (compute-heavy, token-parallel) then joins the rotation.
4. **Backpressure.** A stage's input queue past a high watermark throttles upstream
   admission; a queue that stops draining past a timeout is a dead-stage signal →
   recovery.
5. **Fairness.** Round-robin across active streams by default; per-stream priority is
   a later knob. No head-of-line monopoly.

**Correctness hazard (gated).** Because stages are shared across all K streams,
**per-stream KV must be strictly isolated** at every stage — a token of stream A must
never read stream B's KV. The cross-stream KV-contamination test (token-for-token
correctness across all K streams vs a single-stream reference) is a **hard gate**.
**[LIVE]** the cross-stream KV-isolation test passed on a live llama-3.1-8b N=2 run;
occupancy gain is characterized in §5.

### 3.4 The recovery model: token-history-is-truth replay

**Principle.** KV is derived; the token sequence is truth; recovery is replay. No KV
is ever migrated or backed up. The **durable stream token-history** (the in-flight
stream's token-IDs — cheap, IDs not tensors) lets any stage's KV be reconstructed by
re-prefilling its block over that history.

**`RecoveryPolicy`** is selected by the gateway/driver per event:

- **Reassign (default, hot path).** Preconditions: spares hold block weights
  pre-staged (disk/host-RAM warm, VRAM-cold); the in-window work is a VRAM-load +
  CUDA-graph capture for **one block** (tens of seconds), never a download, never the
  whole model. The gateway re-points the two affected transport edges
  (i−1 → spare → i+1); the spare runs embed→block-*i* prefill over the stream's
  existing context, populating **only the replaced block's cache**. Downstream caches
  are untouched — re-streaming through them would double-append and corrupt them.
  Prefill is token-parallel, hence fast.
- **Rebuild (fallback).** Falls through when no warm spare is available, simultaneous
  losses exceed spare count, or there is weight-version/quant skew between the dead
  node and the spare (cross-version activations are garbage). Accepts a pipeline
  break.

**Proactive vs reactive.** The **proactive drain is the primary path.** On the ~2-min
spot pre-warning, the dying node signals the driver *while still alive*; the in-flight
stream is migrated to the warm spare gracefully, with no dropped token. The
**reactive** path (after a hard kill) is the fallback: an edge-supervision timeout
fires, the token mid-traversal at the dead stage is lost, and the stream is re-driven
from the retained history (replay). No corruption results, because KV is rebuilt, not
resumed from a partial.

**Cost model.** Recovery cost scales with **context length × death-depth** (how many
stages must replay). The danger zone is high eviction rate × long contexts; this is
the concrete form of the thrash risk and the number to watch.

**Multi-stream interaction.** An eviction stalls only the streams *through the dead
stage*; the rest keep flowing. The driver holds the affected streams, re-prefills, and
resumes — partial recovery, not a fleet freeze. This is why §3.3 and §3.4 are
co-designed.

**Node state machine.**
```
provisioning → staging (weights→disk) → warm (disk-staged, VRAM-cold)
   → loading (VRAM-load + graph capture) → active (serving block)
   → draining (eviction warning) → dead
```

### 3.5 Warm spare + self-replenish

The warm pool has two dials: warm-spare **count** and the **spot-vs-on-demand mix**
of the floor. v1.0 default is **1 warm spot spare, 0 on-demand floor**, configurable
up on both. A spare is one warm spot instance, disk-staged with the full model,
VRAM-cold, loading the dead block on demand — **not pre-pinned to a block**, because
eviction is roughly random across instances (at higher counts, boundary/highest-risk
blocks can be kept VRAM-warm).

Recovery runs at **two timescales**: the warm spare takes the dead block in
**seconds** (keeps serving); SkyPilot backfills a replacement instance in **minutes**,
which becomes the *new* warm spare, restoring the safety margin. The spare covers the
gap; the provisioner refills the slot. **[LIVE]** consuming the spare auto-fired a
real `sky launch` of a replacement box (host-side replenish watcher), verified live.

### 3.6 Load-on-promotion reshape

A spare is generic — it holds *all* slices on local NVMe but loads none into VRAM
until promoted. On promotion it **re-execs into the dead rank's slice** (loads exactly
that block's weights NVMe→VRAM). This is what lets one undifferentiated spare cover
any rank's death, and it is what makes warm re-slicing possible: a fit that OOMs by a
hair can be re-cut (e.g. 3-way → 4-way) **on the same running boxes** from on-disk
weights (~8 min) rather than a cold relaunch (~40 min). **[LIVE]** the 4-way re-slice
from disk was performed live; **[CPU-PROVEN]** the generic-spare re-exec into a dead
rank's slice is validated off-GPU.

### 3.7 Durable-STAR catastrophic recovery (designed)

Single-stage recovery covers the common case; catastrophic and orchestrator-failure
recovery is a **durable star, not a peer-to-peer mesh**:

- The committed token history mirrors continuously to a **Cloudflare Durable Object**
  (survives *all* GPU boxes dying — the token history is the crown jewels).
- Nodes heartbeat to the control plane (a star: nodes↔control-plane, **not**
  nodes↔nodes). No peer gossip, no leader election, no consensus.
- The recovery loop re-provisions lost stages; each replays from the durable history
  (the proven replay-rebuild).
- The **driver is resumable** — its state lives in the DO, so a fresh driver resumes
  if its box dies.

This is the standard "durable coordinator + fungible workers" pattern (a Kubernetes
control plane, not a pod mesh). A peer mesh was **rejected**: it adds N²-monitoring
and split-brain risk for no benefit, and re-introduces exactly the
permissionless/decentralized machinery the fork deleted. **[DESIGNED]** the
`/internal/{heartbeat,stale,recovery}` + token-mirroring endpoints exist and are
sim-tested; the control plane is not deployed.

---

## 4. Implementation

### 4.1 Fork lineage and what was stripped

Cairn is a **hard fork of `leyten/shard`** (Apache-2.0, SGLang-based pipeline
inference), pinned to a specific upstream commit and explicitly *not* kept mergeable.
The fork's three hardest problems vanish under Cairn's owned-fleet assumption and were
deleted:

| Shard's hard problem | Why it vanishes for Cairn |
|---|---|
| WAN transport / NAT hole-punching | Owned LAN inside one VPC. |
| Volunteer-node activation-leakage privacy | We own every node; no malicious middle node. |
| Decentralized payments / permissionless join | Single operator, controlled fleet. |

Also stripped: per-node identity issuance / keyed-handshake scaffolding (a pre-shared
`SHARD_PSK` suffices for an owned fleet). **Kept and re-pointed:** the contiguous-layer
split + block runtime, the supervised edges, and the encrypted no-pickle wire (§3.2).

The **net-new** is the multi-stream scheduler (§3.3) and the control/fleet glue
(gateway + Durable Objects). Everything else is fork or adopt — the "new shape" budget
is spent only where no existing tool provides it.

### 4.2 SGLang per-block wrap

SGLang provides kernels, attention, paging, and quantization — Cairn does **not**
rebuild any of these. Each node loads one contiguous block and exposes
`forward(hidden, kv_meta) → hidden`, holding per-node KV for its block across all
streams. The wrap is in-process around SGLang's `ModelRunner` (the offline `Engine`
proved finicky for this; two `ModelRunner`s per process conflict, so a shared runner
is used). Splitting a model required: 0-based per-block layer re-indexing, a per-block
KV cache, and a path-β slice that mutates the live parallel groups so each box loads
only its layer range from disk into VRAM rather than the whole checkpoint.

### 4.3 SkyPilot provisioning

SkyPilot is adopted at the **instance-provisioning + per-instance spot-recovery**
layer (managed-jobs-per-node), *not* SkyServe — SkyServe's abstraction is *N identical
replicas*, whereas Cairn's topology is a *pipeline of heterogeneous-block nodes*.
Block assignment, topology, and edge re-stitch are owned by Cairn's controller.
Operational discipline that proved load-bearing: scoped least-privilege IAM (no admin
keys; destructive actions gated on a `cairn=true` tag and region-locked); EC2-API
teardown verification (`sky status` is not ground truth — an INIT-wedged cluster can
fail to terminate while still billing); a tag-scoped kill switch; in-region weight and
image caches (S3 for the ~294 GB checkpoint, ECR for the base image) so GPU boxes never
HF-pull gigabytes on the clock; and a hash-pinned `requirements.lock` (every package
sha256-pinned) for reproducible, drift-free installs.

### 4.4 sm_120 / Blackwell specifics

The headline runs target **workstation Blackwell (sm_120, RTX PRO 6000 = AWS g7e)**,
which at the time of writing had **no stock-upstream working path** for the DSA/MoE-FP4
model family: DeepGEMM (the GEMM library DeepSeek-V4 / GLM-5.2 architectures require)
hard-refuses sm_120, and mainline SGLang's Triton attention for several of these models
emits PTX (`tile::gather4`, `.shared::cluster`) that exists only on sm_100/sm_90a. The
path forward was a community sglang fork (`0xSero`) that runtime-patches the official
Blackwell image with sm_120 FlashMLA kernels. Bringing a frontier model up on this
chip required clearing ~11 numbered walls — load path under layer-split, per-layer norm
gating, the V4 4D wire shape, an SWA-translation-cache global-layer-0 assumption, the
MoE GEMM-config init ordering (the FP8 linear method binds its GEMM function at
construction, so the config must be initialized *before* `ModelRunner` construction) —
ending at a clean FP8 GEMM on a loading model. A notable cost-saving lesson: **wall
#11 was the wrong checkpoint, not a kernel bug** — the int8 V4 variant has no sm_120
MoE kernel and fell through to an unquantized path; the FP8 repack
(`sgl-project/DeepSeek-V4-Flash-FP8`) was the fix, diagnosed *byte-level for $0 of GPU
time* via a shape probe + HF-header reads. The general rule the project adopted: match
a community fork's *exact* checkpoint variant, not just the model family.

### 4.5 Repository layout

```
cairn/
  fork/        # forked + stripped Shard: block runtime (SGLang wrap) + LAN transport
  scheduler/   # the multi-stream scheduler — the core build
  control/     # CF Workers gateway + Durable Objects fleet state
  infra/       # SkyPilot configs, instance bootstrap, weight-staging, caching
  configs/     # model-as-config: gpt-oss-120b.yaml, deepseek-v4-flash.yaml, glm-5.2.yaml, …
  bench/       # benchmark harness + gate-artifact tests (correctness, interruption, crypto)
  docs/        # vision, spec, handoff, this paper
```

---

## 5. Evaluation

### 5.1 Plan

The evaluation answers one question: **does Cairn beat the on-demand baseline on
cost/token at a given reliability** — i.e. does the minimal reassign-first primitive
deliver on-demand-grade continuity at spot cost, and stay competitive with the richer
recovery mechanisms of prior systems on *their* models. The plan has three tiers
(detailed methodology in `report/benchmark-plan.md`, authored separately):

1. **Comparability (head-to-head on published models).** GPT-NeoX-20B (SpotServe's
   exact 20B, Apache-2.0) to compare Cairn's fixed-block+replay recovery against
   SpotServe's reparallelize+migrate on cost/token and tail latency (target: ≤
   SpotServe's published **54% of on-demand**); Llama-3.1-8B (KevlarFlow's exact) to
   compare **MTTR** under matched failure injection.
2. **Proof.** gpt-oss-120b — the mechanism on a current open model.
3. **Headline.** DeepSeek-V4-Flash and GLM-5.2 (NVFP4/FP8, g7e/Blackwell) — the
   frontier flex the 2024 papers could not run, where the split is *mandatory*. No
   published baseline exists at that size by choice; the small-model head-to-heads
   carry the comparison.

The gate metrics: cost/token vs on-demand, MTTR, per-stage occupancy,
split-correctness (token-for-token vs single-reference greedy), and the
induced-interruption timeline (warning → load → re-prefill → resume).

### 5.2 Preliminary live results

All numbers below are **[LIVE]** on real GPUs unless tagged otherwise.

**Setup (headline live run).** DeepSeek-V4-Flash FP8 (~600 GB on disk, 43 layers)
split four ways across 4× g7e single-GPU Blackwell spot instances (Ohio), in one VPC.

**Distributed frontier decode.** The 4-way split produced coherent, correct output:
a definition of "cairn," `17×24 = 408`, and a full ~990-word essay — confirming the
split is correct end-to-end on the headline model.

**Single-stream hot-swap recovery (the central result).**

| Scenario | MTTR | Correctness | Drops |
|---|---|---|---|
| Pre-warmed L4, single box | **0.023 s** | bit-identical | 0 |
| V4 on Blackwell, mid-generation | **0.93 s** | bit-identical | 0 |
| V4 on Blackwell, mid-essay (drained at token 43, resumed at 44) | **3.26 s** | bit-identical, no visible seam | 0 |

The tail stage was drained mid-generation; the warm spare was re-stitched; generation
resumed bit-identical (output sha matched the baseline across the token boundary) with
**zero dropped or duplicated tokens**, and the endpoint never returned an error.

**Recovery is position-aware.** Proactive drain (the ~2-min spot pre-warning) carries
the dying stage's identity → graceful migration *before* death. The reactive
(abrupt-SIGKILL) half-open-timeout fallback is proven on the tail.

**Multi-stream throughput / occupancy.**

| Concurrent streams (K) | Throughput |
|---|---|
| 1 | 6.1 tok/s |
| 2 | 12.4 tok/s |
| 4 | 24.5 tok/s |
| 8 | 24.9 tok/s |

Throughput scales near-linearly to the fleet ceiling (~4× occupancy gain before
saturation at this N), the empirical demonstration that multi-streaming rescues the
1/N single-stream utilization.

**Self-replenishing pool.** Consuming the spare auto-fired a real `sky launch` of a
replacement spot box (host-side replenish watcher), restoring the warm margin —
verified live.

**Warm re-slicing.** A 3-way fit that OOM'd by a hair (~94–95 GiB/box) was re-cut to
4-way (~74 GiB/box) **on the same running boxes** from on-disk weights (~8 min) vs a
cold ~40 min relaunch.

**Safe spot operations.** Scoped least-privilege IAM (no admin keys); `cairn=true`-
tagged teardown verified via the EC2 API as ground truth; kill-switch by tag.

### 5.3 Proven in simulation / on CPU (pending live confirmation)

Mechanically the same code as the live paths, validated bit-identical off-GPU:

- **Any-position recovery** (entry / middle / tail): 7/7 bit-identical. **[CPU-PROVEN]**
- **Multi-stream recovery** (K streams swap together onto one spare, all replayed):
  3/3. **[CPU-PROVEN]**
- **Load-on-promotion reshape** (generic spare re-execs into the dead rank's slice).
  **[CPU-PROVEN]**
- **Catastrophic / control-plane recovery** (durable token history mirrored off-box;
  star topology) — endpoints exist and are sim-tested. **[DESIGNED]**
- Roughly **110 automated tests** green (Python scheduler/fit/recovery + TS control
  plane) before any GPU spend, including the §3.3 cross-stream KV-isolation check on
  the mock runtime.

### 5.4 Open / pending measurements (TODO)

- **TODO — Live multi-stream recovery:** swap K concurrent streams together onto one
  spare on a live fleet (CPU-proven; needs a warm spare, ~30 min to warm a fresh box).
- **TODO — Entry/mid live drains:** proactive drain of the entry and a middle stage on
  a live fleet (tail proven live; entry/mid CPU-proven).
- **TODO — Clean multi-death:** ≥2 simultaneous losses (needs ≥2 warm spares; parked).
- **TODO — Interior reactive death:** abrupt kill of a middle box requires heartbeats
  (the Path-2 control plane) to identify *which* interior node died; proactive drain
  covers it today.
- **TODO — Head-to-head numbers:** cost/token vs SpotServe (target ≤54% on-demand) on
  GPT-NeoX-20B; MTTR vs KevlarFlow on Llama-3.1-8B; tokens-dropped distributions. The
  measurement harness exists; the runs are pending.
- **TODO — Cost/token first-class output:** per-request blended spot cost and the
  spot-vs-on-demand savings number (currently computed offline, not yet a primary
  metric of a live run).
- **TODO — Endurance/thrash:** multi-hour serving through *real* (not induced) spot
  reclaims at target blended cost; thrash bound (reload-overhead / total-spend < Y%).

---

## 6. Related Work

Cairn executes a **published, peer-reviewed thesis** — fault-tolerant LLM serving on
preemptible instances is an active area — and the contribution is the specific
recovery recipe and its productization at frontier scale, not the idea. (Full
head-to-head methodology in `report/benchmark-plan.md`.)

**SpotServe** (CMU, ASPLOS '24) — the direct precedent: "first distributed LLM serving
system on preemptible instances," 54% cheaper than on-demand. It recovers by **dynamic
reparallelization** (Kuhn-Munkres matching to re-shard the whole model) plus
**token-granularity KV migration** inside the grace window. Cairn differs by keeping
blocks **fixed** and swapping only the dead block onto a pre-staged spare, rebuilding
just that block's KV by replay — a simpler primitive with blast radius 1, *unproven vs
SpotServe's throughput at scale; that is the bet, not a given*.

**Petals** (NeurIPS '23) — Cairn's *architecture*: transformer blocks across nodes,
surviving node loss by rerouting to another node holding the same block, at
internet/volunteer scale. Cairn shares the block-split but targets a low-RTT
single-VPC homogeneous spot fleet for production multi-stream throughput, and uses
*replay of token-history-as-truth* rather than rerouting stored activations.

**KevlarFlow** (Jan 2026) — the latest: decoupled MP init + dynamic rerouting +
**background KV replication**, a 20× MTTR cut. Cairn avoids continuous KV replication
entirely (it would swamp the wire and covers only graceful eviction, never a hard
crash), trading replication bandwidth for replay cost at recovery time.

**Helix and other fleet/scheduling systems** target heterogeneous-GPU placement and
throughput; they do not provide pipeline multi-streaming with a reassign-first spot
failover as a single co-designed unit.

**Adjacent:** DéjàVu (KV streaming), AnchorTP (state daemons + KV recompute), LMCache,
RLBoost.

**Positioning.** The shared regime — sharding one model across many GPUs and surviving
node loss — is **table stakes**, not the differentiator; it is what makes the
comparison fair. Cairn's "distinctly more" is the **combination of axes**: a model
*big enough that the split is mandatory* (frontier scale on commodity single-GPU spot)
× production multi-stream throughput × a *minimal* recovery primitive (replay the dead
block only, no reparallelization, no KV replication) × per-block SGLang with an own
ChaCha20 wire that **dodges the NCCL hang-on-failure** every prior system fights ×
2026-era model/HW (4-bit DSA on workstation Blackwell) the 2024 papers could not use.

---

## 7. Limitations & Honest Scoping

1. **The contribution is the failover layer, not the model.** Cairn rebuilds no
   kernels, attention, paging, or quantization — those are SGLang. It is a
   systems/ops reliability layer. The honest framing is "≈ on-demand reliability at
   spot cost," and the failover *unlocks* the 3–4× spot discount rather than adding
   cost beyond one standing warm spare.

2. **Spare warm-up latency (~20–40 min) is the principal operational ceiling.**
   Loading the ~294 GB checkpoint into a fresh spare is slow. One swap is invisible;
   a *burst* of swaps faster than spares can warm would eventually exhaust the pool.
   This is the single most important caveat. The fix is operational (snapshots /
   pre-baked AMIs / NVMe pre-stage / a deeper standing pool), not research — but it is
   what most separates "a demo that recovers" from "a service you trust." Reclaim
   correlation (one AWS event taking many same-AZ instances) makes this worse and is
   the reason warm-pool sizing must be eviction-rate-keyed.

3. **The durable control plane is designed, not deployed.** Today the driver box is a
   single point of failure for in-flight requests; the Cloudflare DO durable
   token-history and resumable driver are specified and sim-tested but not live. Until
   then, catastrophic (all-boxes / orchestrator) recovery is unproven on real hardware.

4. **Single-fleet happy path.** The live results are a single fleet in one region. Not
   yet demonstrated: multi-AZ/region spot hunting under churn, sustained-load soak,
   chaos testing (random multi-reclaims), multi-tenancy (per-tenant auth/quotas/
   isolation, which extends the §3.3 KV-isolation requirement to a hard cross-tenant
   boundary), autoscaling N and stream capacity with demand, and first-class cost
   accounting.

5. **Interior reactive death is not yet covered live.** An abrupt kill of a *middle*
   node needs heartbeats (the Path-2 control plane) to identify which interior node
   died; today the proactive drain covers interior nodes, and only the *tail*'s
   reactive path is live-proven.

6. **"Beat SpotServe" is a higher bar than "beat on-demand."** SpotServe and
   KevlarFlow are strong systems from elite groups. The literature de-risks "cheap +
   survivable"; out-performing full reparallelization or KV-replication with the
   minimal primitive is the open bet, and the head-to-head numbers (§5.4 TODO) are not
   yet in hand.

7. **Regime-bound by construction.** Cairn loses, and says so, where a model fits an
   affordable single/multi-GPU box — inter-stage overhead and imperfect bubble-fill
   make a split less FLOPS-efficient than one box. The claim holds only in the
   VRAM-gap + spot-availability-gap regime.

---

## 8. Conclusion

Cairn shows that the affordable, deep supply on the spot market — cheap single-GPU
instances — can be turned into reliable serving capacity for a model far too large to
fit any one of them, by treating reclaims as a *routine* 1/N event rather than a
catastrophe. The enabling ideas are deliberately minimal: a strict contiguous-layer
split with a per-block transport that **eliminates cross-node NCCL** (and with it the
hang-on-failure that breaks every naive approach); a **token-history-is-truth recovery
model** in which KV is derived and recovery is replay onto a pre-staged warm spare;
and a **multi-stream scheduler** that fills the pipeline bubble to keep the split
economical.

The preliminary live evidence is encouraging: a 600 GB frontier model served across
four single-GPU Blackwell spot instances, with mid-generation hot-swap recovery in
**0.93 s and 3.26 s** — bit-identical, zero drops — multi-stream throughput scaling
**6.1 → 24.9 tok/s**, and an automatically self-replenishing warm pool **[LIVE]**. The
remaining recovery matrix is bit-identical in simulation **[CPU-PROVEN]**, and the
durable catastrophic-recovery path and head-to-head benchmarks are the clear next
work. We are explicit that the technical risk — *can the minimal primitive keep a
frontier model serving through preemption?* — is substantially retired, while the
product risk (durability, warm-up latency, multi-tenancy, scale hardening) is not.

The artifact is open source so that others can run their own fault-tolerant spot
serving in production, and so that the central bet — that a minimal reassign-first
primitive stays competitive with reparallelization and KV-replication at frontier
scale — can be tested, reproduced, and pushed on by the community.

---

## Appendix A — Evidence ledger (claim → tier → source)

| Claim | Tier | Source |
|---|---|---|
| 4-way split serves coherent V4-Flash output (cairn def, 17×24, Rome essay) | LIVE | `report/achievements.md`; session 2026-06-28 |
| Single-stream MTTR 0.023 s (pre-warmed L4) / 0.93 s / 3.26 s, bit-identical, 0 drops | LIVE | `report/achievements.md`; `bench/` recovery artifact |
| Multi-stream throughput 6.1 → 12.4 → 24.5 → 24.9 tok/s (K=1/2/4/8) | LIVE | `report/achievements.md`; session 2026-06-28 |
| Self-replenish auto-fires `sky launch` | LIVE | `report/achievements.md` |
| Warm 3→4-way re-slice from disk (~8 min) | LIVE | `plan/history.md` (2026-06-27); `report/achievements.md` |
| 3-way OOMs ~94–95 GiB / 4-way fits ~74 GiB | LIVE | `report/achievements.md` (the keystone) |
| Crypto self-test at boot; tampered frame resets edge | LIVE | `plan/history.md` (2026-06-19) |
| Cross-stream KV-isolation passes (llama-3.1-8b N=2) | LIVE | `plan/history.md` (2026-06-24) |
| Contiguous-split == single-ref greedy, token-for-token | LIVE (small) / CPU-PROVEN (ref) | `plan/history.md` (2026-06-21) |
| Any-position recovery (entry/mid/tail) 7/7 bit-identical | CPU-PROVEN | `report/achievements.md` |
| Multi-stream recovery 3/3 bit-identical | CPU-PROVEN | `report/achievements.md` |
| Load-on-promotion reshape (generic spare re-exec) | CPU-PROVEN | `report/achievements.md` |
| Durable-STAR catastrophic recovery (DO history, resumable driver) | DESIGNED | `plan/history.md` (2026-06-22); endpoints sim-tested |
| Head-to-head cost/token vs SpotServe (≤54%), MTTR vs KevlarFlow | TODO | `report/benchmark-plan.md` (planned) |
| Live multi-stream swap; entry/mid live drains; multi-death; endurance/thrash | TODO | `report/productization.md` |
