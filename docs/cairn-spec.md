# Cairn — Technical Specification

> **Doc 2 of 3** (Vision & Roadmap → **Spec** → Agent Handoff).
> Reads on top of `cairn-vision-roadmap.md` — thesis, scope/regime, non-goals,
> economics, and roadmap live there and are not repeated.
> **Status:** v0.1 · covers the v1.0–v1.2 architecture as one whole (multi-stream
> and recovery are specced up front because v1.0 must be built to accommodate
> them). v1.3+ is named, not specced. Working name **Cairn** (trademark gate open).

---

## 0. Invariants — the spine every component reads from

Written before any feature. These are the contract between phases; nothing below
may violate one without changing this list deliberately.

1. **Contiguous-layer split.** The model is cut into contiguous blocks of layers,
   one block per node. **No tensor-parallel across nodes, no expert-parallel.** For
   MoE, every layer's experts and its router live wholly on one stage — routing is
   intra-stage, never a cross-node dispatch. (This is what keeps MoE simple here.)
2. **Token history is the source of truth; KV-cache is derived.** Recovery is
   replay. In-flight stream token-IDs are retained durably (§7). No KV is ever
   backed up or migrated.
3. **Model is configuration.** The scheduler fits *any* contiguous-layer model
   (dense or MoE) to the joined VRAM. No model-specific code paths. The two named
   targets (§3) are two configs.
4. **The control plane is out of the per-token hot path.** The decode loop runs
   entirely inside the AWS VPC. Cloudflare does admission, orchestration, durable
   stream state, and (later) billing — never a per-token round trip.
5. **One recovery interface.** `RecoveryPolicy`: reassign default, rebuild
   fallback, selected per event (§5).
6. **Owned fleet, single VPC, LAN.** No WAN, no NAT, no untrusted nodes. This is
   the assumption that deletes Shard's two hardest problems (transport cost-center,
   activation-leakage privacy) — see §6, §8. If it is ever relaxed, both return.
7. **One ingress.** All external requests pass the gateway's validator before
   admission. One door.

---

## 1. System architecture

Three planes. The hot path is wholly inside plane 3.

```
PLANE 1 — CLIENT            PLANE 2 — CONTROL (Cloudflare)        PLANE 3 — DATA (AWS spot, one VPC)
 API request          ──►   Gateway (Workers): auth, admit,   ──► Entry node ─► … ─► Tail node ─► sample
 (OpenAI-compatible)        validate, set up route                (embed+L0..)   (..lm_head)        │
 token stream         ◄──   Fleet state (Durable Objects):    ◄──        ▲───────── per-token ──────┘
                            registry · topology · health ·            (sampled token re-enters entry;
                            durable stream token-history             whole loop stays on the VPC LAN)
                            ── drain / retry / reassign ──►
```

- **Entry node drives generation** (Shard's "head" role): holds the embedding +
  the first block, injects the prompt, and runs the decode loop. The sampled token
  from the **tail node** (which holds the final block + `lm_head`) returns to the
  entry node to begin the next traversal. Both the forward chain and the tail→entry
  return are in-VPC.
- **Cloudflare is control-plane only.** It is not in the per-token loop (would add a
  CF↔AWS hop per token — fatal to latency). It admits requests, orchestrates the
  fleet, and holds durable stream state for recovery.

**Request lifecycle:** admit (gateway, validate) → assign to a pipeline + register
stream in DO → **prefill** (prompt processed once, KV populated at every stage) →
**decode rotation** (stream joins the multi-stream loop, §4) → tokens streamed back
through the gateway → on completion, stream dropped from DO and KV freed at every
stage.

---

## 2. Components & ownership

| Component | Build / Fork / Adopt | Responsibility |
|---|---|---|
| Per-node block runtime | **Fork Shard** (SGLang-based) | Load one contiguous block; `forward(hidden_states, kv_meta) → hidden_states`; hold per-node KV for its block across all streams. Kernels/attention/paging/quant are SGLang — not ours. |
| Inter-stage transport | **Fork Shard, strip to LAN** | Stream the activation tensor stage→stage; supervised edges (health, timeout, reconnect). NAT/hole-punch/relay **deleted** (§6). |
| **Multi-stream scheduler** | **BUILD — the core net-new** | Interleave K streams through the pipeline to fill the bubble; admission control; per-stage queues; backpressure (§4). |
| Gateway / router | **Build** (CF Workers) | One ingress: auth, validate, admit, route-setup, token streaming. OpenAI-compatible API. |
| Fleet controller + state | **Build** (CF Durable Objects) | Registry, topology, health, durable stream token-history, the drain/retry/reassign loop (§5, §7). |
| Spot provisioning + recovery infra | **Adopt SkyPilot/SkyServe** | Provision/replace spot instances, multi-region hunt, on-demand fallback. Confirm current feature set at build. |
| Weights distribution | HF + cache | Pull a node's block weights from Hugging Face; cache on instance/EBS for fast re-stage. |

The "new shape" exception (Build-vs-Adopt doctrine) is spent only on the
multi-stream scheduler + the gateway/fleet glue. Everything else is fork or adopt.

---

## 3. Model fit & block granularity (N)

**Inputs:** target model (layer count, per-layer VRAM incl. experts for MoE,
embedding + final-norm + `lm_head` sizes), quant, the set of joined GPUs
(heterogeneous VRAM allowed), and **target K** (concurrent streams — drives KV
headroom).

**Output:** a contiguous layer-block assignment per node where, for each node:

```
block weights  +  KV headroom (K streams × max-context × this block's layers)  +  activation buffers   ≤   node VRAM
```

embedding pinned to stage 0, `lm_head` to stage N−1. **The fit reserves KV headroom
for K from day one** — KV is not an afterthought; it is what makes "usable VRAM"
less than card VRAM, and it couples back to N (more KV headroom → fewer layers per
node → more stages).

**Algorithm (greedy, balanced):** walk layers, accumulate onto the current node
until the next layer would exceed its budget, then open the next node. Then
rebalance to **minimise the max-loaded stage** — a single overloaded stage
bottlenecks the whole pipeline (the pipeline runs at the speed of its slowest
stage). Heterogeneous VRAM → uneven blocks (a 48 GB node carries more layers than a
24 GB one); v1.0 uses a homogeneous pool, the algorithm supports mixed for later.

**The two configs (from the vision doc's decisions):**

| | v1.0 proof | Commercial headline |
|---|---|---|
| Model | gpt-oss-120b | Qwen3.5-397B-A17B |
| Footprint | ~63 GB (native MXFP4) | ~397 GB (FP8) / ~200 GB (INT4) |
| Pool | **g6.xlarge** (1× L4, 24 GB) | **g6e.xlarge** (1× L40S, 48 GB) |
| N (weights-only est.) | ≈ 3–4 | ≈ 9–10 (FP8) / ≈ 5 (INT4) |

N above is weights-only; the **real N is larger** once KV headroom for K is
reserved, and is a spec-time calc with the chosen quant and K. The rule is fixed:
**g6/L4 for the shallow v1.0 proof; switch to g6e/L40S for the headline model**,
because 397B on L4 is ~20 hops (latency and per-loss rebuild cost both blow up).
Instance family is a function of N, which is a function of the model.

---

## 4. Multi-stream scheduler (the core net-new)

### 4.1 The problem
In a pipeline of N stages, one token requires a full traversal (stage 0 → … →
N−1 → sample). Autoregressive decode is one token at a time and token *t+1* needs
*t* sampled first, so **a single stream cannot be pipelined** — at any instant one
stage works, N−1 idle. Single-stream utilisation ≈ 1/N. Paying for N GPUs to get
one GPU's throughput. Multi-stream is therefore **not optional**; it is what makes
cost/token defensible.

### 4.2 The mechanism
Run K **independent** streams (different requests). While stream A's token is at
stage 2, stream B's can be at stage 1, stream C's at stage 0. With **K ≥ N**, every
stage is busy on a different stream's token every cycle. This is distributed
continuous batching: vLLM-style continuous batching, but spread *across* a pipeline
split rather than within one node — and nothing off-the-shelf does it across the
split, which is why it is the build.

### 4.3 Design decisions (locked)
1. **Async, queue-driven stages — not lockstep.** Each stage has an input queue of
   `(stream_id, activation)`; it processes FIFO and emits to the next stage's queue.
   The pipeline self-fills as long as ≥ N streams are active. Async tolerates
   per-stage jitter and matches the supervised-edge transport; lockstep would stall
   the whole pipeline on the slowest stage every cycle.
2. **K bounds.** `K_min = N` (saturate). `K_max` = bounded by **per-stage KV
   VRAM** (every stage holds KV for all K streams through its block). Operating K
   tuned within `[K_min, K_max]`. This is the central tension of the scheduler:
   **K must be ≥ N to fill the pipe, and ≤ the VRAM-bounded ceiling.** The fit (§3)
   reserves headroom for the target K.
3. **Admission control.** Admit new streams up to `K_max`; never below `K_min`
   while work exists. A new stream runs its **prefill** (compute-heavy,
   token-parallel — fills KV at every stage) then joins the decode rotation.
4. **Backpressure.** If a stage's input queue passes a high watermark (a stage is
   degraded or downstream is slow), upstream admission throttles. A queue that
   stops draining past a timeout is a dead-stage signal → triggers recovery (§5).
5. **Fairness.** Round-robin across active streams by default; per-stream priority
   is a v1.x knob. No head-of-line monopoly.

### 4.4 Correctness hazard (must be gated)
Stages are shared across all K streams, so the **per-stream KV must be strictly
isolated** at every stage — a token of stream A must never read stream B's KV. A
v1.1 gate artifact explicitly tests for cross-stream KV contamination
(token-for-token correctness across all K concurrent streams vs single-stream
reference). This is the most likely subtle bug in the build.

---

## 5. Recovery model (LOCKED)

### 5.1 Policy
- `RecoveryPolicy`, selected by the gateway per event. **Reassign = default hot
  path. Rebuild = fallback and accepts a pipeline break.** Both built from v1.0.
- **Reassign** preconditions: spares hold block weights **pre-staged** (disk /
  host-RAM warm, VRAM-cold); in-window work = VRAM-load + CUDA-graph capture for
  **one block** (tens of seconds), never a download, never the whole model; gateway
  re-points the two affected transport edges (i−1 → spare → i+1); per-stream KV
  rebuilt on the spare (§5.3).
- **Falls through to Rebuild** when: no warm spare available; simultaneous losses
  exceed spare count; or weight-version/quant skew between the dead node and the
  spare (cross-version activations are garbage — must rebuild).

### 5.2 Warm pool — two dials
- Dial 1 = warm-spare **count**. Dial 2 = **spot-vs-on-demand** mix of the floor.
- **v1.0 default: 1 warm spot spare, 0 on-demand floor.** Configurable up on both.
- A spare = one warm spot instance, disk-staged with the full model, VRAM-cold,
  loading the dead block on demand. **Not pre-pinned to a block** (eviction is
  ~random across instances). At higher counts, keep boundary (highest-risk) blocks
  VRAM-warm.
- Sizing driver = **observed eviction rate**, which is often **correlated** — one
  AWS reclaim can take many same-type/same-AZ instances at once, denting blast
  radius 1/N and able to blow past a 1-spare pool into rebuild. Hedges trade off:
  decorrelate across AZ/type (pay hop latency) vs on-demand floor (costs money).
  Adaptive, rate-keyed sizing = v1.2; static dial = v1.0.

### 5.3 KV-cache on drop — rebuild via replay, never migrate
- Principle (Build doctrine, *replay the log → reconstruct state*): KV is derived;
  the token sequence is the source of truth; recovery is replay.
- The **durable stream token-history** (§7) lets any stage's KV be reconstructed by
  re-prefilling its block over that history.
- Mechanism: the spare runs embed→block-*i* prefill over the stream's existing
  context, populating **only the replaced block's cache**. Downstream caches are
  untouched — re-streaming through them would double-append and corrupt them.
  Prefill is token-parallel → fast.
- Migrate rejected: no copy exists; continuous off-box KV replication would swamp
  the transport (KV is large and grows every token); and it covers only graceful
  eviction, never a hard crash.
- Cost scales with **context length × death-depth** → danger zone = high eviction
  rate × long contexts. This is the concrete form of the thrash risk and the number
  to watch (§10).
- **Multi-stream interaction:** an eviction stalls only the streams *through the
  dead stage*; the rest keep flowing. The gateway holds the affected streams,
  re-prefills, resumes — partial recovery, not a fleet freeze. This is why §4 and §5
  are co-designed.

### 5.4 Node state machine
```
provisioning → staging (weights→disk) → warm (disk-staged, VRAM-cold)
   → loading (VRAM-load + graph capture) → active (serving block)
   → draining (eviction warning) → dead
```
- **Eviction warning (~2 min)** → node `draining` → gateway stops routing new
  tokens to that pipeline position, selects policy, executes reassign (or rebuild).
- **Hard crash (no warning)** → edge-supervision timeout fires → same recovery; the
  token in mid-traversal at the dead stage is lost and re-driven from the gateway's
  retained history (replay). No corruption, because KV is rebuilt, not resumed from
  a partial.

### 5.5 Deferred (v1.3+)
Stage-boundary hidden-state checkpoints (periodic or on the eviction warning) →
resume a rebuild from the nearest boundary instead of embed, trading a little
steady-state bandwidth for cheaper recovery. v1 ships replay-from-embed: correct
and simple.

---

## 6. Transport

**The LAN move is what makes this product simpler than Shard.** In one VPC /
placement group there is no NAT between stages, so **hole-punching and relay
fallback are deleted entirely**, and inter-stage RTT is sub-millisecond. The
activation tensor per token is a single hidden-state vector — KB-scale — far under
the instances' LAN bandwidth (g6 up to 10 Gbps, g6e up to 20 Gbps), so **transport
bandwidth is not the bottleneck** the way it was on Shard's home uplinks. What
remains the bottleneck is per-stage compute and the recovery loop.

Kept from the Shard fork:
- **Wire format** — JSON header + raw tensor bytes, **no pickle** (a hostile frame
  is a parse error, never code execution), sealed with **ChaCha20-Poly1305** under a
  shared `SHARD_PSK`. In a trusted owned VPC the threat model is lighter, but the
  encrypted/authenticated wire is cheap defence-in-depth and already built — keep
  it. Crypto vectors run at boot and fail loud on mismatch (Build doctrine).
- **Supervised edges** — per-edge health, timeouts, fast fault detection,
  reconnect. This is load-bearing: it is how a dead stage is detected and recovery
  triggered. No black-box "broken pipe" — every edge logs its own health.

Config knobs (off by default in-VPC): **activation codec** (fp8/int8 quant of the
activation tensor) — unnecessary at LAN bandwidth, kept as a knob for the AZ-spread
case in v1.2 where hops cross the network.

---

## 7. Control plane & durable state

- **Gateway (CF Workers):** the one ingress. Auth, request validation,
  OpenAI-compatible API, admission, route setup, token streaming back to the client.
  Not in the per-token loop.
- **Fleet state (CF Durable Objects):**
  - **Registry** — who is in the fleet, which block each holds, version/quant.
  - **Topology** — pipeline order (latency-aware when AZ-spread; trivial in one PG).
  - **Health** — heartbeats; the source the recovery loop reads.
  - **Durable stream token-history** — the authoritative copy of each in-flight
    stream's token-IDs (cheap; IDs not tensors), mirrored from the entry node so a
    dead entry node is itself recoverable. This is invariant #2 made concrete.
  - **The drain/retry/reassign loop** — consumes eviction warnings and health
    timeouts, selects `RecoveryPolicy`, orchestrates spare load + edge re-stitch.
- **CF ↔ AWS channel:** control only (assignments, health, recovery commands).
  Provisioning/replacement is delegated to SkyPilot; DO holds the desired-state and
  reconciles.

---

## 8. Security & retention posture

- **Secrets** — HF token (weight pull), AWS credentials (provisioning),
  `SHARD_PSK` (wire) — in a secret store (CF secrets / AWS Secrets Manager), never
  in the repo or an image. Rotatable.
- **Wire** — encrypted + authenticated (kept from fork); crypto self-test at boot.
- **Node identity** — pre-shared `SHARD_PSK` is correct for v1: the fleet is owned
  and known. Shard's note that a *permissionless* swarm needs per-node identities +
  a keyed handshake **does not apply** — we are not permissionless (invariant #6).
- **Tenant isolation** — v1.0 is single-tenant (no isolation needed). When
  multi-tenant lands (v1.3+), the §4.4 per-stream KV isolation requirement extends
  to a hard cross-tenant boundary; specced when it lands, not before.
- **Retention** — commercial track, so a real backend is permitted. **What is
  retained:** in-flight stream token-history (for recovery), dropped on completion.
  No prompt or output content is persisted by default. Later billing needs usage
  *metadata* (token counts), not content. State this honestly in any external copy;
  this is a commercial product and makes no sovereign zero-retention claim.

---

## 9. Milestones & gate artifacts

Gates are pass/fail (vision §7). Each milestone must produce the listed evidence;
nothing advances with a gate open. `/forward-pass` runs between milestones.

### v1.0 — Correctness on controlled spot (single-stream)
Fork onto a SkyPilot-managed g6.xlarge pool, one VPC/placement group, gpt-oss-120b,
single stream, no auth/billing — a benchmark harness.
**Gate artifacts:**
- **Reliability log:** N×g6.xlarge serving gpt-oss-120b, ≥ X/X clean completions.
- **Correctness check:** split output is token-for-token identical to a
  single-reference run on greedy decode (proves the split is correct).
- **Induced-interruption test:** kill a node mid-decode → reassign loads the block
  on the warm spare, re-prefills KV, re-stitches edges, resumes; request completes,
  output uncorrupted. Recovery timeline logged (warning → load → re-prefill →
  resume).
- **Crypto self-test** passes at boot; a tampered frame resets the edge.

### v1.1 — Multi-stream (the core)
**Gate artifacts:**
- **Utilisation:** per-stage GPU occupancy ≥ floor under K concurrent streams.
- **Cost/token** beats the on-demand single-large-instance baseline, in-regime.
- **Cross-stream correctness:** token-for-token correctness across all K streams vs
  single-stream reference (the §4.4 contamination test). Hard gate.

### v1.2 — Fleet hardening + spot economics
**Gate artifacts:**
- **Endurance:** sustained serving through *real* (not induced) spot interruptions
  at target blended cost over a multi-hour run.
- **Thrash bound:** reload-overhead / total-spend < Y%.
- **Eviction telemetry** feeding warm-pool sizing (the rate → dial calibration).

---

## 10. Observability

Per Shard's principle — nothing opaque. The metrics that gate the product:
- **Per-edge health** (latency, throughput, resets) — every edge logs its own.
- **Per-stage occupancy** — the multi-stream utilisation number.
- **cost/token** (blended spot + warm-pool/floor waste) — the economic truth.
- **Eviction rate + correlation** — drives warm-pool sizing.
- **Rebuild cost** (context length × death-depth) and **thrash %** — the §5.3
  danger-zone instruments; the make-or-break numbers.

---

## 11. Deferred — v1.3+ (named, not specced)
Speculative decoding (draft model on the **entry node** — local-draft / remote-
verify; the entry node is the natural "local") · multi-model packing on one fleet ·
Quiver adapter mounting on the served base (a third multiplicative cost lever) ·
billing / auth / multi-tenant + cross-tenant isolation · AZ-spread + adaptive
warm-pool sizing · stage-boundary activation checkpointing.

---

## 12. Spec-time calcs (computations, not decisions)
Resolve at build, before the v1.0 fit:
1. **Exact usable VRAM per card** (24 GB L4, 48 GB L40S) after CUDA context +
   framework overhead + activation buffers → exact weight budget.
2. **KV per stream per block** at the chosen quant → `K_max` per stage → real N.
3. **Micro-batch depth** for prefill interleaving against decode latency.
4. **Warm-pool size** once eviction rate is observed (static for v1.0; the v1.2
   adaptive curve later).
