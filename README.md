# Cairn

> Working name (trademark/domain check is a v0 gate). Status: **mechanism proven live** —
> a distributed-with-recovery frontier-model endpoint runs end-to-end on real spot GPUs.

Hosted inference that serves a single large open model by **pipeline-splitting it
across a fleet of cheap, individually-reclaimable single-GPU spot instances**, kept
utilised by **multi-streaming**, and kept alive through spot interruptions by a
**reassign-first recovery loop**. Frontier-class models on commodity spot — scoped to
the VRAM-and-availability gap where no affordable single instance fits the model.

The mechanism is a hard fork of [`leyten/shard`](https://github.com/leyten/shard)
(pipeline-parallel inference, Apache-2.0), stripped to a LAN-in-one-VPC deployment;
the net-new is the **multi-stream scheduler**, the **warm-spare recovery layer**, and
the control/fleet glue.

## Proven live (2026-06-28)

On a real 4× g7e (Blackwell, sm_120) spot fleet in AWS Ohio, serving
**DeepSeek-V4-Flash FP8** (~600B, 43 layers) split 4 ways:

- **Coherent distributed decode** — correct, long-form answers across the pipeline.
- **Hot-swap recovery** — a stage drained mid-generation, a warm spare re-stitched its
  slice, and the stream resumed **bit-identical, zero dropped tokens** (MTTR 0.93s / 3.26s
  on V4; 0.023s pre-warmed on L4).
- **Multi-stream throughput** — scales ~linearly to the fleet ceiling (6.1→24.5 tok/s,
  1→4 concurrent streams).
- **Self-replenishing pool** — consuming a spare auto-launches a replacement spot box.

See [`report/achievements.md`](report/achievements.md) for the full evidence table and
[`report/productization.md`](report/productization.md) for the gap to a product.

## Architecture (three planes)

- **Plane 1 — Client.** OpenAI-compatible inference API. The agent face *is* the product.
- **Plane 2 — Control (Cloudflare).** Workers gateway (auth, admission, route setup,
  token streaming) + Durable Objects fleet state (registry, topology, health, durable
  stream token-history, the drain/retry/reassign loop). **Never in the per-token hot path.**
- **Plane 3 — Data (AWS spot, one VPC).** A pipeline of cheap single-GPU spot instances,
  each serving one contiguous block of layers. The decode loop runs wholly in-VPC over LAN.

## Documentation

The design is fixed across three docs in [`docs/`](docs/) — build *to* them, don't re-derive:

1. [`docs/cairn-vision-roadmap.md`](docs/cairn-vision-roadmap.md) — thesis, regime/scope, economics, roadmap.
2. [`docs/cairn-spec.md`](docs/cairn-spec.md) — invariants, architecture, fit algorithm, scheduler, recovery model, gates.
3. [`docs/cairn-agent-handoff.md`](docs/cairn-agent-handoff.md) — build plan, repo structure, cadence, hard rules.

Start with the spec's **§0 invariants** and keep them open while building.

## Status

**Mechanism proven live** (distributed decode + hot-swap recovery + multi-stream +
self-replenish, all on real spot GPUs — see above). The remaining work is largely
productionization, not research: live-confirm the full recovery matrix (multi-stream
swap, entry/mid/multi-death drains), head-to-head benchmarks vs prior art, the control
plane for catastrophic durability, and faster spare warm-up. Gap analysis:
[`report/productization.md`](report/productization.md).

> The full operator-facing README (per agent-handoff §13 — deploy the control plane,
> launch the data plane, model-as-config, benchmark harness, env/secrets) is the next
> consolidation step now that there's a runnable system to document.
