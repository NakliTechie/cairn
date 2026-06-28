# Cairn — Prior-Art Comparison & Head-to-Head Benchmark Plan

_Authored 2026-06-28. Companion to [`plan/prior-art.md`](../plan/prior-art.md) (positioning)
and [`report/achievements.md`](achievements.md) (live results). This doc is the
**evaluation contract**: the cited prior-art numbers, exactly where Cairn differs from each
system, the experimental protocol for an apples-to-apples comparison, and a results-table
template with their published numbers pre-filled and empty columns for Cairn's live numbers._

> **Honest framing (carried from `plan/prior-art.md`).** Cairn is **not** first to distributed
> LLM serving on spot / survivable-node-loss inference. The contribution is the **specific
> recovery recipe + regime** (a frontier model that fits no single commodity GPU, served on
> production spot, recovered by warm-spare reassign-first + bit-identical token-history replay,
> kept busy by multi-stream occupancy), not the idea of sharding a model. The comparison below
> is built to be *fair to strong systems from elite groups*, not to flatter Cairn.

---

## (a) Prior-art comparison table — cited published numbers

Columns: what it does · eval models · eval hardware · fault-tolerance/recovery approach ·
the headline numbers **the paper itself reports**. Numbers are quoted from the sources linked
in each row; where a paper reports a *range* it is given as a range.

| System | What it does | Eval models | Eval hardware | FT / recovery approach | Reported headline numbers |
|---|---|---|---|---|---|
| **SpotServe** (CMU/PKU/CUHK, ASPLOS '24) | First distributed LLM serving system on **preemptible (spot)** instances. Dynamically re-parallelizes the whole model on topology change; migrates instances via bipartite (Kuhn-Munkres) matching; stateful inference recovery inside the grace window. | OPT-6.7B (25 GB), GPT-NeoX-20B (74.5 GB), LLaMA-30B (111.8 GB); min 4/12/16 GPUs resp. | **AWS g4dn.12xlarge (4× NVIDIA T4 / instance)**; real **12-hour** g4dn spot availability trace, two 20-min segments replayed | **Reparallelize** (re-shard whole model, restart serving) + **KV migration** at token granularity within the ~30 s grace window; bipartite matching minimizes migration cost. | **54%** of on-demand-only monetary cost (spot $1.9/hr vs on-demand $3.9/hr) for **<18% avg-latency / <90% P99** increase. P99 tail latency **2.4–9.1×** better than best prior serving systems; on LLaMA-30B **1.34–2.43×** (vs reparallelize) and **2.14–9.13×** (vs reroute). [arXiv 2311.15566](https://arxiv.org/abs/2311.15566) · [github.com/Hsword/SpotServe](https://github.com/Hsword/SpotServe) |
| **Petals** (BigScience, NeurIPS '23) | BitTorrent-style **decentralized / volunteer** collaborative inference + fine-tuning. Transformer blocks spread across nodes; survive node loss by **rerouting stored activations** to another node holding the same block. Internet/WAN scale. | BLOOM-176B, Llama-2-70B | Heterogeneous **consumer / volunteer GPUs over the internet** (high-RTT WAN) | Client holds a **chain of servers**; on a server drop, reroute the request to another server holding that block (activations re-sent). No central operator. | Single-batch interactive decode: **up to ~6 steps/s (Llama-2-70B)**, **~1 step/s (BLOOM-176B)**; **up to 10× faster than offloading** (offloading: ≥5.5 s/token RAM, ~22 s/token SSD for BLOOM-176B). [arXiv 2209.01188](https://arxiv.org/abs/2209.01188) · [arXiv 2312.08361](https://arxiv.org/abs/2312.08361) |
| **KevlarFlow** (Jan 2026) | **Resiliency-focused** serving: decoupled model-parallel init + dynamic traffic rerouting + **background KV-cache replication** so a partial failure becomes a near-instant migration, not a restart. | **Llama-3.1-8B**, 4-stage pipeline parallel (one stage / node) | **8- and 16-node**, each **1× NVIDIA A10 (24 GB)**, 1 Gbps Ethernet, nodes across **4 datacenter locations** (geo-distributed LB group); **injected node failures** | Background KV replication to other nodes in the LB group → on failure, partially-served requests continue on a live replica; dynamic rerouting. | **MTTR 10 min → ~29–35 s (≈20×)**. Under failure: avg latency **3.1×**, P99 latency **2.8×**, **avg TTFT 378.9×**, **P99 TTFT 574.6×** better than SOTA. Steady-state overhead **~2.3–4.0% avg / 2.8–3.6% P99**. [arXiv 2601.22438](https://arxiv.org/abs/2601.22438) |
| **Helix** (CMU Thesys, ASPLOS '25) | High-throughput serving over **heterogeneous** GPUs+network. Formulates placement+routing as **max-flow / MILP**; jointly optimizes model placement and request scheduling. (Throughput, not FT.) | LLaMA-class models | Heterogeneous clusters **24–42 GPU nodes** | None (not a fault-tolerance system). | Serving throughput **up to 3.3×**, prompt latency **−66%**, decode latency **−24%** vs prior approaches. [arXiv 2406.01566](https://arxiv.org/abs/2406.01566) · [github.com/Thesys-lab/Helix-ASPLOS25](https://github.com/Thesys-lab/Helix-ASPLOS25) |
| **SkyServe / SkyPilot serving** (UC Berkeley, 2024) | Cross-region/cross-cloud serving on **spot**, dynamic **spot+on-demand mix**, autoscale by QPS, decorrelate preemptions by spreading across regions/clouds. **Replica-level** (each replica is a whole model copy), not a single split model. | Llama-2-70B, OPT-6.7B | A10G, T4, V100, A100-80GB across regions/clouds | **Stateless replica** recovery: preempted replica is replaced; over-provision cheap spot, scale down on-demand when spot returns. No in-request state recovery. | **~50%** cost savings (up to **44%** vs pure on-demand); spot **2–12×** cheaper than on-demand. P50/P90/P99 latency **2.6×/3.1×/2.7×**. Availability single-zone→multi-region **29.9%→95.8% (A100)**, **68.2%→99.2% (V100)**. Replica provision **~183 s** (> 2-min spot warning). [arXiv 2411.01438](https://arxiv.org/abs/2411.01438) · [SkyServe blog](https://blog.skypilot.co/introducing-sky-serve/) |
| **FlexGen** (Stanford et al., ICML '23) | Single-GPU **high-throughput offloading** (GPU+CPU+disk), 4-bit weight/KV compression. Throughput-max for batch/offline, not serving or FT. | OPT-175B (and family) | **One 16 GB GPU** (e.g. T4) + CPU + disk offload | None (single box; no node loss model). | **~1 token/s** for OPT-175B on a single 16 GB GPU; **up to ~100×** the throughput of prior single-GPU offloading systems. [arXiv 2303.06865](https://arxiv.org/abs/2303.06865) |
| **DeepSpeed-Inference / -MII / -FastGen** (Microsoft) | Production single-/multi-GPU serving: tensor-parallel within a node, blocked KV cache, continuous batching, Dynamic SplitFuse, fused kernels. Throughput/latency, **no spot/FT story**. | BLOOM-176B, GPT-family, Llama, 37k+ models | Multi-GPU **within one node** (NVLink TP) | None for node loss (it is the efficient single-node baseline Cairn is *out of regime* against). | DeepSpeed-FastGen: **up to 2.3×** effective throughput, **~2×** lower avg latency, **up to 3.7×** lower token-level tail latency vs vLLM. [arXiv 2401.08671](https://arxiv.org/abs/2401.08671) · [arXiv 2207.00032](https://arxiv.org/abs/2207.00032) |
| **DéjàVu** (MSR/ETH, 2024) | Pipeline-parallel serving with **KV-cache streaming**: prompt/token disaggregation (bubble reduction), microbatch swapping (memory), **state replication** for FT. | OPT-family | Multi-GPU pipeline | **KV-cache streaming + replication** to a backup so a failed pipeline stage resumes from streamed state. | Addresses pipeline bubbles, memory over-provisioning, and long failure-recovery; FT via KV streaming/replication (qualitative recovery-time gains; see paper). [arXiv 2403.01876](https://arxiv.org/abs/2403.01876) · [github.com/msr-fiddle/dejavu](https://github.com/msr-fiddle/dejavu) |

**Cross-cutting read.** The fault-tolerant systems split into two recovery philosophies:
**(1) move the state** — KV migration (SpotServe), KV streaming/replication (KevlarFlow, DéjàVu),
activation rerouting (Petals); **(2) replace the replica** (SkyServe, stateless). Cairn is a
**third option: keep no state at all and replay the token log** onto a pre-staged warm spare —
the only one where recovery is *bit-identical by construction* rather than by careful state copy,
and the only one with **no cross-node collective** (NCCL/TP) to hang on a death.

---

## (b) Cairn's differentiation, per system

The shared regime — "shard one model across many GPUs and survive node loss" — is **table stakes**,
not the differentiator (this framing is locked in `plan/prior-art.md`). The distinctness is the
*combination* of: contiguous-layer split with **no cross-node NCCL**, **warm-spare reassign-first**
recovery with **bit-identical token-history replay**, **multi-stream occupancy** on one fixed
pipeline, and a **single-VPC owned fleet** (not a permissionless mesh).

- **vs SpotServe** — Same problem (serving on spot), opposite recovery primitive. SpotServe
  **reparallelizes** (re-shards the *whole* model and restarts serving) and **migrates KV** within
  the grace window. Cairn keeps blocks **fixed**, swaps only the **one dead block** onto a warm
  spare, and **rebuilds just that block's KV by replay** (never migrates KV). Blast radius is 1/N,
  not the whole model; recovery is bit-identical, not a careful state copy. **Scale gap:** SpotServe
  evaluated ≤30B on 4× T4 (sharding for economics — the model *fits* a few GPUs); Cairn's regime is
  a model that fits **no single commodity GPU** (V4-Flash 671B), where the split is **mandatory**.
  *The open bet (stated, not assumed): does the minimal reassign+replay primitive stay competitive
  with SpotServe's full reparallelization on cost/token at tail latency?*

- **vs Petals** — Same contiguous-block split, opposite network + recovery substrate. Petals is a
  **permissionless volunteer swarm over WAN**; recovery is **reroute stored activations** to another
  volunteer holding the block. Cairn is a **single-VPC, owned, homogeneous spot fleet** (sub-ms RTT,
  no NAT/hole-punch, no activation-leakage privacy problem), recovery is **token-history replay onto
  a warm spare**, and the target is **production multi-stream throughput**, not single-batch
  interactive latency on a swarm. Petals' BLOOM-176B is comparable scale but at volunteer-internet
  latency (~1 step/s); Cairn targets LAN occupancy.

- **vs KevlarFlow** — Closest in spirit (resiliency-first, pipeline-parallel, injected failures),
  opposite state policy. KevlarFlow **replicates KV in the background** to a live node so a failure
  is a near-instant migration. Cairn **keeps no KV backup at all** — token history is the only
  durable state, KV is rebuilt by replay. That trades a small recovery-compute cost (re-prefill the
  dead block) for **zero steady-state replication bandwidth** and **bit-identical correctness** with
  no replica-divergence risk. KevlarFlow evaluated **Llama-3.1-8B on A10s** (a model that fits one
  GPU; split for resilience); Cairn's V4-Flash split is **mandatory**. MTTR is the head-to-head
  number: KevlarFlow **~29–35 s**; Cairn's **live single-stream V4 MTTR is 0.93 s / 3.26 s**
  (pre-staged warm spare, one-block reload) and **0.023 s** pre-warmed on L4 — a different recovery
  regime because only one block reloads, not a model-parallel re-init.

- **vs Helix** — Helix is a **throughput/placement** optimizer (max-flow/MILP over heterogeneous
  GPUs) with **no fault-tolerance**. It is a useful **scheduler-quality** reference for the
  multi-stream occupancy axis, not a recovery comparison. Cairn's scheduler is simpler (async
  queue-driven contiguous pipeline) but its differentiator is the *combination with recovery*, which
  Helix does not address.

- **vs SkyServe / SkyPilot serving** — SkyServe operates at the **replica** granularity: each
  replica is a **whole model copy** on spot, recovered by **replacement** (stateless). Cairn
  operates at the **block** granularity: **one model split across the fleet**, recovered by
  **reassigning the dead block** while the request survives. SkyServe **cannot serve a model that
  exceeds a single replica's box** — exactly Cairn's regime. Cairn *adopts* SkyPilot for spot
  provisioning/replacement (the self-replenish loop), so this is layering, not rivalry: Cairn is the
  in-request survival layer SkyServe lacks.

- **vs FlexGen / DeepSpeed** — Out of regime, included as **boundary baselines**. FlexGen is
  single-GPU offload throughput (no serving, no FT); DeepSpeed is efficient **single-node** TP
  serving (no spot, no node-loss). They define the "single-box wins, don't pretend otherwise" edge
  from the vision doc §2 — Cairn only claims its wedge where the model **exceeds an affordable single
  instance**. DeepSpeed-on-one-box is the explicit thing Cairn differentiates *from*, not *beats*.

**The genuinely-Cairn one-liner.** Per-block SGLang + own ChaCha20-Poly1305 wire (no pickle, **no
cross-node NCCL** → dodges the collective-hang-on-failure every TP system fights), the
**token-history-is-truth / recovery = replay-the-dead-block-only** invariant, single-VPC commercial
edge, multi-stream occupancy on a fixed pipeline, and 2026 frontier model/HW (V4-Flash FP8 on
Blackwell g7e spot) the 2024 papers could not run.

---

## (c) Head-to-head benchmark protocol

### Goal
An **apples-to-apples** comparison on the axes that matter, honest about what can and cannot be
reproduced. We do **not** have SpotServe's / KevlarFlow's exact clusters; we therefore separate
**(i) head-to-head on their exact models** (where a fair direct comparison is possible) from
**(ii) regime claims on our own frontier model** (where no published baseline exists by choice).

### Models — three tiers (from `plan/prior-art.md`, carried verbatim in intent)
1. **Comparability (head-to-head vs the papers):**
   - **GPT-NeoX-20B (Apache-2.0)** — SpotServe's exact 20B → reproduce SpotServe's setup (real spot
     trace, on-demand baseline, P99 + avg latency + cost/token). Target: **≤ SpotServe's 54%**
     on-demand cost at comparable tail latency, with Cairn's reassign+replay instead of
     reparallelize+migrate.
   - **Llama-3.1-8B** — KevlarFlow's exact → reproduce their **failure-injection** (1- and 2-node
     kills) → compare **MTTR distribution**.
   - OPT-6.7B (SpotServe's, research-only license) — internal benchmark only, never product/demo.
   - BLOOM-176B (Petals', RAIL) — optional, internal-only.
   - **Skip** LLaMA-30B (research-only/leaked license).
2. **Proof:** Llama-3.1-8B on commodity GPUs — the mechanism on a current open model.
3. **Headline:** **DeepSeek-V4-Flash FP8 (671B)** on g7e/Blackwell — the frontier flex the 2024
   papers could not run (this is the model already live; see `report/achievements.md`).

### Hardware fairness caveats (state these in the paper)
- SpotServe used **g4dn / T4**; Cairn uses **g6 / L4** (T4's successor). Run Cairn on g6/L4 **and
  cite their T4 numbers as reference**; a literal g4dn/T4 re-run is the "exactly-exactly" fallback if
  a reviewer pushes.
- KevlarFlow used **A10 (24 GB), 1 Gbps, 4-datacenter** geo-spread; Cairn is **single-VPC LAN**
  (sub-ms). This is a **regime difference, not a bug** — Cairn's whole thesis is the owned single-VPC
  LAN. Report Cairn's MTTR as a *different recovery regime* (one-block reload vs MP re-init), do not
  claim the geo-distributed delta as ours.
- Where exact-cluster reproduction is infeasible, report **our absolute numbers + their published
  numbers side by side** and label the row "**reference, not reproduced**." Never silently rescale
  someone's number.

### Metrics to measure (the apples-to-apples set)
| Metric | Definition | Why it's the comparison axis |
|---|---|---|
| **MTTR distribution** | Wall-clock from induced death → first correct resumed token. Report **min / median / P95 / max** over ≥20 induced reclaims per scenario (not a single number). | Direct head-to-head vs KevlarFlow (29–35 s) and SpotServe's grace-window recovery. |
| **Dropped / duplicated tokens** | Diff resumed stream vs single-reference greedy run, token-for-token. | Cairn's bit-identical claim is binary — must be **0 dropped, 0 duplicated**, every run. |
| **Recovery success rate under induced reclaims** | (clean completions) / (induced reclaims). Separate **reassign-path** vs **rebuild-fallback** rates. | Reliability gate (spec §9, vision §7). |
| **Aggregate throughput vs concurrency** | tok/s at K = 1, 2, 4, 8, … up to saturation. | The multi-stream occupancy axis; compare scaling shape, not just peak. |
| **Per-stage occupancy** | GPU-busy fraction per stage under load. | Whether multi-stream actually fills the bubble (spec §10 floor). |
| **Time-to-first-token (TTFT)** | Admission → first token, steady-state and under failure. | KevlarFlow's headline FT axis (378.9× / 574.6×). |
| **Cost per 1M tokens** | Blended (spot + warm-pool/on-demand floor + rebuild waste) ÷ tokens. | The economic truth; SpotServe's 54% is the bar. |
| **Spot-vs-on-demand % savings** | Cairn blended cost ÷ on-demand single-large-instance cost, in-regime. | Headline economics vs SpotServe 54% / SkyServe ~50%. |
| **Steady-state overhead** | Throughput/latency cost of the recovery machinery when nothing fails. | Cairn's "no KV replication" should give ~0 steady-state overhead vs KevlarFlow's 2.3–4.0%. |

### Induced-reclaim methodology
- **Proactive drain** (models the ~2-min spot warning): deliver the drain signal to the targeted
  stage; the gateway carries the dying stage's identity, reassigns to the warm spare before death.
  Measure warning→load→re-prefill→resume timeline. (Live-proven on the tail; see achievements.)
- **Reactive death** (models hard reclaim): `SIGKILL` the stage process (or `aws ec2
  terminate-instances` on the tagged box for the real-reclaim variant); edge-supervision timeout
  fires → recovery. Measure timeout→detect→reassign→resume.
- **Positions:** entry, middle, tail — run each independently (any-position recovery is CPU-proven
  7/7; live-confirm per position).
- **Multi-death:** ≥2 simultaneous kills with ≥2 warm spares (tests reassign-vs-rebuild fallover).
- **Concurrency under failure:** repeat induced reclaims at K = 1, 2, 4, 8 to confirm only the
  streams *through the dead stage* stall (partial recovery, not a fleet freeze — spec §5.3).
- **Sample size:** ≥20 induced events per (scenario × position) cell → a real distribution, not an
  anecdote. Log each event's full timeline.

### Warm-up & fairness handling
- **Discard** the first N tokens / first stream of every run as warm-up (CUDA graph capture, page
  pool fill); report steady-state only, and say so.
- **Pre-stage** the warm spare disk-warm/VRAM-cold for the **reassign** path (the spec'd default);
  separately report a **cold-spare** number so the warm-pool assumption is explicit, not hidden.
- **Same prompts, same greedy decode, same max-context** across Cairn and the reference run for the
  correctness diff. Fix seeds; greedy decode for the bit-identical check.
- **Cost accounting** includes the warm-pool / on-demand floor and rebuild waste — never quote the
  raw spot price as cost/token (that would flatter Cairn and the spec forbids it, vision §6).
- **Report what we did not reproduce.** Any row that cites a paper's cluster we couldn't replicate is
  labeled "reference," and the fairness caveat travels with the number.

### What Cairn already has live (baseline to slot in — from `report/achievements.md`)
- **Single-stream hot-swap MTTR:** **0.93 s** and **3.26 s** on V4-Flash (Blackwell, mid-essay),
  **bit-identical, 0 dropped tokens**; **0.023 s** pre-warmed on L4.
- **Multi-stream throughput:** **6.1 → 12.4 → 24.5 → 24.9 tok/s** at K = 1 / 2 / 4 / 8
  (~4× occupancy gain before saturation).
- **Self-replenish:** consuming a spare auto-fired a real replacement `sky launch` (live).
- **Any-position + multi-stream recovery:** CPU-proven bit-identical (7/7, 3/3); live-confirm pending.

---

## (d) Results-table template — their numbers pre-filled, Cairn columns empty

> Fill the **Cairn (live)** columns from benchmark runs. "n/a" = the system does not report /
> does not address that axis. "ref" = published number cited as reference, not reproduced on our
> cluster. Keep every cited number's source URL in the footnotes.

### Table 1 — Recovery (head-to-head on the FT systems)
| Metric | SpotServe | KevlarFlow | DéjàVu | Petals | SkyServe | **Cairn (live)** |
|---|---|---|---|---|---|---|
| Recovery model | reparallelize + KV migrate | KV replicate + reroute | KV stream + replicate | reroute activations | replace replica | reassign block + replay |
| Eval model | OPT-6.7B / NeoX-20B / LLaMA-30B | Llama-3.1-8B | OPT-family | BLOOM-176B / Llama-2-70B | Llama-2-70B / OPT-6.7B | **DeepSeek-V4-Flash 671B** |
| Split mandatory? (model > 1 GPU) | no (≤30B) | no (8B) | no | yes (176B, WAN) | no (replica = whole model) | **yes (671B)** |
| MTTR (median) | within ~30 s grace window | ~29–35 s | "fast" (qual.) | reroute latency (WAN) | ~183 s (replica reprovision) | _0.93 s / 3.26 s (V4); 0.023 s (L4)¹_ |
| MTTR distribution (min/P95/max) | n/a (point) | ~29–35 s band | n/a | n/a | n/a | _to measure (≥20 events/cell)_ |
| Dropped tokens | migrate (lossless intent) | migrate (lossless intent) | replicate | re-sent | request replaced | _0 (bit-identical)¹_ |
| Duplicated tokens | n/a | n/a | n/a | n/a | n/a | _0 (bit-identical)¹_ |
| Bit-identical to single-ref? | not claimed | not claimed | not claimed | not claimed | not claimed | _yes, greedy (7/7 CPU; live ✓)¹_ |
| Recovery success rate (induced) | high (paper) | high (paper) | — | — | — | _to measure (reassign vs rebuild split)_ |
| Steady-state FT overhead | n/a | 2.3–4.0% avg / 2.8–3.6% P99 | replication BW | n/a | n/a | _expect ~0 (no KV replication)_ |
| Cross-node collective (NCCL/TP)? | yes | yes (MP) | yes (PP) | no | n/a | **no (per-block, ChaCha20 wire)** |

### Table 2 — Throughput & latency (occupancy / serving)
| Metric | SpotServe | KevlarFlow | Helix | FlexGen | DeepSpeed-FastGen | SkyServe | **Cairn (live)** |
|---|---|---|---|---|---|---|---|
| Aggregate throughput | — | — | up to **3.3×** vs prior | ~1 tok/s OPT-175B (1×16 GB) | up to **2.3×** vs vLLM | — | _6.1/12.4/24.5/24.9 tok/s @ K=1/2/4/8²_ |
| Throughput vs concurrency curve | n/a | n/a | n/a | n/a | continuous batching | n/a | _measured (above); extend K_ |
| Per-stage occupancy | n/a | n/a | max-flow optimized | n/a | n/a | n/a | _to measure (≥ floor)_ |
| Avg latency (under failure) | <18% incr. @ 54% cost | **3.1×** better | n/a | n/a | ~2× lower (no-fail) | n/a | _to measure_ |
| P99 latency | **2.4–9.1×** better | **2.8×** better | n/a | n/a | up to **3.7×** lower tail | **2.7×** better | _to measure_ |
| TTFT (avg / P99, under failure) | n/a | **378.9× / 574.6×** | prompt lat −66% | n/a | n/a | n/a | _to measure_ |

### Table 3 — Cost / spot economics
| Metric | SpotServe | SkyServe | FlexGen | **Cairn (live/target)** |
|---|---|---|---|---|
| Cost vs on-demand-only | **54%** (i.e. 46% saved) | **~50%** (up to 44% saved) | n/a (single box) | _target ≤ 54%; to measure_ |
| Spot discount cited | $1.9 vs $3.9/hr (~2×) | spot **2–12×** cheaper | n/a | _to measure (g6/L4, g7e)_ |
| Cost per 1M tokens | — | — | — | _to measure (blended, incl. floor + rebuild)_ |
| Availability gain (multi-region) | n/a | 29.9→95.8% (A100) | n/a | _single-VPC by design; warm-pool dial_ |
| Self-replenish after reclaim | reparallelize | replace replica | n/a | _live ✓ (auto sky launch)²_ |

**Footnotes / sources.**
¹ `report/achievements.md` (live, 2026-06-28). ² Same.
SpotServe: [arXiv 2311.15566](https://arxiv.org/abs/2311.15566).
Petals: [arXiv 2209.01188](https://arxiv.org/abs/2209.01188), [arXiv 2312.08361](https://arxiv.org/abs/2312.08361).
KevlarFlow: [arXiv 2601.22438](https://arxiv.org/abs/2601.22438).
Helix: [arXiv 2406.01566](https://arxiv.org/abs/2406.01566).
SkyServe: [arXiv 2411.01438](https://arxiv.org/abs/2411.01438), [blog](https://blog.skypilot.co/introducing-sky-serve/).
FlexGen: [arXiv 2303.06865](https://arxiv.org/abs/2303.06865).
DeepSpeed-FastGen/Inference: [arXiv 2401.08671](https://arxiv.org/abs/2401.08671), [arXiv 2207.00032](https://arxiv.org/abs/2207.00032).
DéjàVu: [arXiv 2403.01876](https://arxiv.org/abs/2403.01876).

---

## Use-this-doc notes
- **The honest gate (from `plan/prior-art.md`):** the question is not "are we first" (no), but
  "do we beat the on-demand baseline on cost/token at a given reliability — matching SpotServe's
  54% + KevlarFlow's MTTR, on the **same** models." Tables 1–3 are how that gets answered.
- **Do not over-claim MTTR.** Cairn's sub-second MTTR is a **different recovery regime** (one-block
  reload from a pre-staged warm spare, no model-parallel re-init), not a like-for-like beat of
  KevlarFlow's full MP re-init + reroute. Always state the regime alongside the number.
- **Fill order:** Llama-3.1-8B failure-injection (MTTR distribution vs KevlarFlow) → GPT-NeoX-20B
  spot-trace (cost/token + tail vs SpotServe) → V4-Flash regime numbers (own baseline). The first
  two are the head-to-heads; the third is the flex.
