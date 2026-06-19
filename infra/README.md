# infra/ — data-plane provisioning (SkyPilot)

Provisions the single-GPU spot pool in one VPC/placement group, stages weights to
instance/EBS, boots the block runtimes + scheduler, registers nodes with the control
plane (handoff §4). **Adopt SkyPilot** — do not reimplement provisioning / multi-region
hunt / on-demand fallback (spec §2).

**Open seam to validate at build (handoff §3, do NOT assume):** SkyServe assumes *N
identical replicas* — our topology is a *pipeline of heterogeneous-block nodes*, which
is not that. Current research (plan/history.md) points at SkyPilot's **Managed Jobs API**
(it runs pipelines + auto-recovers from spot preemption per task) for the
provisioning + per-instance spot-recovery layer, with Cairn's controller owning
block-assignment + topology + re-stitch. Confirm the exact primitive on a live trial
before committing. **Status: not started.**
