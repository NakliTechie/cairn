# Cairn

> Serve a frontier-class open model on a fleet of cheap, individually-reclaimable
> single-GPU spot instances — and keep the token stream alive, bit-identical, through
> spot reclaims.

Cairn pipeline-splits one large open model **by layers** across N cheap single-GPU AWS
spot instances, keeps the fleet busy by **multi-streaming**, and keeps it alive through
spot interruptions with a **reassign-first recovery layer**: when a stage's box is
reclaimed mid-generation, its slice is re-stitched onto a pre-staged **warm spare** and
the committed **token history is replayed** under a fresh sequence — so generation
resumes exactly where it left off, with no dropped or duplicated tokens. The pool then
**re-provisions itself** back to full strength.

Frontier-class models on commodity spot — scoped to the VRAM-and-availability gap where
no affordable single instance fits the model. The honest contribution is the
**failover/recovery layer**, not the model: the failover is what makes the 3–4× spot
discount usable for serving.

## Proven live (2026-06-28)

On a real 4× `g7e` (Blackwell, sm_120) spot fleet in AWS, serving **DeepSeek-V4-Flash
FP8** (~600B, 43 layers) split 4 ways:

- **Coherent distributed decode** — correct, long-form answers across the pipeline.
- **Hot-swap recovery** — a stage drained mid-generation, a warm spare re-stitched its
  slice, and the stream resumed **bit-identical, zero dropped tokens** (tail MTTR 0.93s
  abrupt / 3.26s proactive on V4; 0.023s pre-warmed on L4). Recovery is
  position-independent (entry/middle/tail proven bit-identical; tail live-confirmed,
  entry/middle GPU-confirmation pending).
- **Multi-stream throughput** — scales ~linearly to the fleet ceiling (6.1 → 24.5 tok/s,
  1 → 4 concurrent streams).
- **Self-replenishing pool** — consuming a spare auto-launches a replacement spot box.

Full evidence table: [`report/achievements.md`](report/achievements.md).

## Quickstart

You need an AWS account (scoped key), [SkyPilot](https://skypilot.readthedocs.io/), and a
Hugging Face token. Then:

```sh
cp infra/secrets.env.example infra/secrets.env   # fill in HF_TOKEN + AWS creds
bash infra/skypilot/bringup.sh                    # launch fleet + serve + tunnel
# defaults: DeepSeek-V4-Flash FP8, 4-way, 6 nodes (4 active + 2 spares), us-east-2
```

Then open `http://localhost:8000/` (built-in chat page) or call the OpenAI-compatible API
at `http://localhost:8000/v1`. Teardown: `infra/skypilot/nuke.sh --force`.

The full guide — every topology/recovery knob, model-as-config, the drain-sentinel test
hook, teardown with EC2-API verification — is in
[`docs/operator-README.md`](docs/operator-README.md).

## Architecture (three planes)

- **Plane 1 — Client.** OpenAI-compatible inference API; the agent face *is* the product.
- **Plane 2 — Control.** Gateway (auth, admission, route setup, token streaming) +
  durable fleet state (registry, topology, health, durable token-history, the
  drain/retry/reassign loop). **Never in the per-token hot path.**
- **Plane 3 — Data (AWS spot, one VPC).** A pipeline of cheap single-GPU spot instances,
  each serving one contiguous block of layers. The decode loop runs wholly in-VPC over
  LAN — **no cross-node NCCL** (which would hang when a spot node vanishes); Cairn wraps
  SGLang per-block with its own encrypted wire and recovery (blast radius 1/N).

## How it compares

Cairn is a distinct, simpler recovery primitive at frontier scale — it neither migrates
KV state (SpotServe, KevlarFlow) nor replaces whole replicas (SkyServe): it keeps **no**
recovery state and **replays the token log** onto a warm spare. Detailed positioning and
the head-to-head benchmark plan vs SpotServe / Petals / KevlarFlow / Helix:
[`report/benchmark-plan.md`](report/benchmark-plan.md).

## Documentation

- [`docs/operator-README.md`](docs/operator-README.md) — run it yourself (bring-up, knobs, teardown).
- [`docs/cairn-vision-roadmap.md`](docs/cairn-vision-roadmap.md) — thesis, scope, economics, roadmap.
- [`docs/cairn-spec.md`](docs/cairn-spec.md) — invariants, fit algorithm, scheduler, recovery model.
- [`docs/paper-draft.md`](docs/paper-draft.md) — the systems write-up (design, evaluation, related work).
- [`report/achievements.md`](report/achievements.md) — what's proven, with evidence.
- [`report/productization.md`](report/productization.md) — the gap from "works" to "product."

## Status & honesty

**The mechanism is proven live** (distributed decode + hot-swap recovery + multi-stream +
self-replenish, on real spot GPUs). It is **not production-ready**: remaining work is
largely productionization, not research — GPU-confirm the non-tail recovery positions and
multi-death, run the head-to-head benchmarks, deploy the control plane for catastrophic
durability, and shorten spare warm-up. See
[`report/productization.md`](report/productization.md).

Claims in this repo are tagged by evidence tier (live-proven / CPU-proven / designed);
we don't publish numbers we haven't measured live.

## License

Apache-2.0 — see [`LICENSE`](LICENSE). Cairn is a hard fork of
[`leyten/shard`](https://github.com/leyten/shard) (Apache-2.0; provenance and the
keep/strip ledger in [`fork/UPSTREAM.md`](fork/UPSTREAM.md)). Models are served as
configuration and carry their own licenses (the headline models are Apache-2.0 / MIT).

> "Cairn" is a working name.
