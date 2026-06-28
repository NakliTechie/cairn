# Cairn

> Serve a frontier-class open model on a fleet of cheap, individually-reclaimable
> single-GPU spot instances — and keep the token stream alive, bit-identical, through
> spot reclaims.

Cairn pipeline-splits one large open model **by layers** across N cheap single-GPU AWS spot
instances, keeps the fleet busy by **multi-streaming**, and survives spot interruptions with
a **reassign-first recovery layer**: when a box is reclaimed mid-generation, its slice is
re-stitched onto a pre-staged **warm spare** and the committed **token history is replayed** —
so generation resumes exactly where it left off, no dropped tokens. The pool then
**re-provisions itself**. The honest contribution is the failover layer, not the model: it's
what makes the 3–4× spot discount usable for serving.

## Headline numbers (proven live, 2026-06-28)

On a real `g7e` (Blackwell, sm_120) spot fleet running **DeepSeek-V4-Flash FP8** (671B-param
MoE, 43 layers, ~294 GB checkpoint), 4-way split:

| Metric | Result |
|---|---|
| **Hot-swap recovery (MTTR)** | **0.93 s** abrupt kill · **3.26 s** proactive drain — **bit-identical, zero dropped tokens** (0.023 s pre-warmed on L4) |
| **Multi-stream throughput** | **6.1 → 12.4 → 24.5 → 24.9 tok/s** at 1 / 2 / 4 / 8 streams (~4× by 4 streams, then fleet ceiling) |
| **Distributed decode** | coherent, correct long-form output across the 4-stage pipeline |
| **Self-replenish** | a consumed spare auto-launches + warms a replacement spot box |

Recovery is position-independent (entry/middle/tail bit-identical; **tail live-confirmed**,
entry/middle CPU-proven). Self-replenish provisioned + warmed a real spare live; recovery
*onto* a replenished spare is fixed in code and pending a live re-validation. Full evidence:
[`report/achievements.md`](report/achievements.md).

## Quickstart

Needs an AWS account (scoped key), [SkyPilot](https://skypilot.readthedocs.io/), and a Hugging
Face token.

```sh
cp infra/secrets.env.example infra/secrets.env   # fill in HF_TOKEN + AWS creds
bash infra/skypilot/bringup.sh                    # launch fleet + serve + tunnel
```

Then open `http://localhost:8000/` (built-in chat page) or call the OpenAI-compatible API at
`http://localhost:8000/v1`. Teardown: `infra/skypilot/nuke.sh --force`. Full guide (knobs,
model-as-config, teardown verification): [`docs/operator-README.md`](docs/operator-README.md).

## How it works

Three planes — an OpenAI-compatible **client** API, a **control** plane (auth, fleet state,
the drain/retry/reassign loop) that's never in the per-token hot path, and a **data** plane of
single-GPU spot boxes each serving one contiguous block of layers. The decode loop runs wholly
in-VPC over LAN with **no cross-node NCCL** (which would hang when a spot node vanishes) — Cairn
wraps SGLang per-block with its own encrypted wire and recovery, blast radius 1/N. Details:
[`docs/cairn-spec.md`](docs/cairn-spec.md) · [`docs/paper-draft.md`](docs/paper-draft.md).

It neither migrates KV state (SpotServe, KevlarFlow) nor replaces whole replicas (SkyServe) —
it keeps **no** recovery state and **replays the token log** onto a warm spare. Positioning +
benchmark plan: [`report/benchmark-plan.md`](report/benchmark-plan.md).

## Roadmap & open directions

- **Faster pool re-arm after a swap (warm-AMI + FSR).** The swap is already sub-second; the
  remaining latency is *refilling* the pool. Baking weights+image into a per-region AMI cuts a
  replacement spare's warm-up from ~30 min to ~5–10 min, and **Fast Snapshot Restore** to
  ~3–5 min — so the **next spare comes up fast right after a hotswap**, keeping the fleet
  continuously protected under sustained churn instead of leaving a ~30 min gap.
- **Further experiments:** push past the ~25 tok/s ceiling (speculative decoding, fatter
  slices), multi-AZ spot hunting, heartbeat-based interior death detection, catastrophic
  durability (Cloudflare DO + resumable driver), model-as-config breadth, autoscaling.

Full detail: [`docs/roadmap.md`](docs/roadmap.md). Gap-to-product analysis:
[`report/productization.md`](report/productization.md).

## Status

**Mechanism proven live; not production-ready.** Remaining work is largely productionization,
not research (GPU-confirm the non-tail recovery positions, head-to-head benchmarks, control
plane, faster replenish). Claims are tagged by evidence tier (live-proven / CPU-proven /
designed) — we don't publish numbers we haven't measured live.

## License

Apache-2.0 ([`LICENSE`](LICENSE)). Cairn is a hard fork of
[`leyten/shard`](https://github.com/leyten/shard) (Apache-2.0; ledger in
[`fork/UPSTREAM.md`](fork/UPSTREAM.md)). Models are served as configuration and carry their own
licenses. "Cairn" is a working name.
