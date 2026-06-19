# Cairn

> Working name (trademark/domain check is a v0 gate). Status: v0.1 — scaffold.

Hosted inference that serves a single large open model by **pipeline-splitting it
across a fleet of cheap, individually-reclaimable single-GPU spot instances**, kept
utilised by **multi-streaming**, and kept alive through spot interruptions by a
**reassign-first recovery loop**. Frontier-class models on commodity spot — scoped to
the VRAM-and-availability gap where no affordable single instance fits the model.

The mechanism is a hard fork of [`leyten/shard`](https://github.com/leyten/shard)
(pipeline-parallel inference, Apache-2.0), stripped to a LAN-in-one-VPC deployment;
the net-new is the **multi-stream scheduler** and the control/fleet glue.

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

Scaffold only. The first build chunk is **v1.0** (correctness on controlled spot,
single-stream) — see [`docs/cairn-agent-handoff.md`](docs/cairn-agent-handoff.md) §7.

> The full operator-facing README (per agent-handoff §13 — deploy the control plane,
> launch the data plane, model-as-config, benchmark harness, env/secrets) is a **v1.0
> deliverable**, written once there's a runnable system to document. This stub is the
> placeholder until then.
