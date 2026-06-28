# How far is Cairn from a product?

_Snapshot: 2026-06-28. Honest gap analysis from "the mechanism works live" to "someone other
than us can rely on it."_

## The short answer

The **hard, novel core is done and proven live**: a frontier model on reclaimable spot with
invisible failover. What's left is mostly **productionization and operability** — durability,
multi-tenancy, automation, and the unglamorous reliability engineering — not more research.

Rough framing: **the technical risk is retired; the product risk is not.** Call it ~60–70%
of the way to a *credible open-source artifact people run*, and earlier-stage as a *managed
service* (which may never be the goal — see below).

## What "done" means depends on the target

The stated end-state (see `docs/cairn-objectives.md`) is an **open-source research artifact +
consulting upside**, not a SaaS. That changes the bar. Two different finish lines:

### Target A — open-source artifact + benchmarks + paper (the stated goal)
**Gap: small-to-medium.** What's needed:
1. **GPU-confirm the non-tail recovery positions.** Recovery is already proven
   position-independent on CPU (bit-identical 7/7: entry/middle/tail), and the **tail** is
   live-confirmed on GPU (4×, proactive + reactive). What's pending is converting
   entry/middle from CPU-proven → live-on-GPU, plus live multi-stream swap and multi-death.
   _~1–2 GPU sessions._
2. **Head-to-head benchmarks** vs the prior art (SpotServe / Petals / KevlarFlow) on their
   exact models — the "on-demand-grade reliability at ~spot cost" number, with MTTR and
   tokens-dropped distributions. _The measurement harness exists; needs runs._
3. **Reproducible launch** — one-command bring-up, documented env/secrets, a clean operator
   README. _Most pieces exist as scripts; needs consolidation._
4. **Write-up** — the systems contribution (the failover layer), honestly scoped.

### Target B — a managed BYOK endpoint others depend on (probably *not* the goal)
**Gap: large.** This is where the real productization work lives:

| Area | Status | What's missing |
|---|---|---|
| **Catastrophic durability** | Designed, sim-tested | Deploy the Cloudflare control plane (DO durable token-history, resumable driver); today the driver box is a single point of failure for in-flight requests. |
| **Interior reactive death** | Proactive-only live | Heartbeats (Path 2) to identify *which* middle node died on an abrupt kill, not just the tail. |
| **Multi-tenancy** | Single API key, single fleet | Per-tenant auth, quotas, isolation, admission control under load. |
| **Autoscaling** | Fixed N + warm pool | Scale N and stream capacity with demand; the throughput ceiling (~25 tok/s here) needs to grow with the fleet. |
| **Observability** | Logs + ad-hoc probes | Metrics, dashboards, alerting, SLOs (MTTR budgets, drop-rate). |
| **Cost accounting** | Computed, deprioritized | Real per-request cost, the spot-vs-on-demand savings number as a first-class output. |
| **Spare warm-up latency** | ~20–40 min (load 294 GB) | The biggest operational gap: a fresh spare takes too long to warm, so deep reclaim storms could outrun replenishment. Needs faster staging (snapshots / pre-baked AMIs / NVMe pre-stage) and/or a deeper standing pool. |
| **Self-replenish networking** | Bug found + fixed (design) | A replenished spare launches as a *separate* SkyPilot cluster → its own isolated security group, so the recovery wire (7777-7780) couldn't reach it (see below). Fixed by pinning the fleet + spare to one shared named SG; not yet re-validated live. |
| **Hardening at scale** | Single-fleet happy path | Sustained-load soak tests, chaos testing (random multi-reclaims), back-pressure, partial-failure modes. |

### Discovered live (2026-06-28): self-replenish SG isolation broke recovery onto a replenished spare

The first end-to-end test of the **self-replenish** path surfaced a real bug. An auto-provisioned
warm spare warmed fully (model loaded, listening on `:7777`) but **could not participate in recovery**,
because `replenish-watcher.sh` launches each replacement spare as its **own SkyPilot cluster** — which
SkyPilot gives a **distinct, isolated security group**. The recovery wire is *bidirectional* across ports
**7777-7780** (stage-in `:7777`, tail-sink `:7779`, spare-sink/announce `:7780`), so SG isolation broke
**both** halves:

1. The spare's announce dial to the driver's `:7780` hung in TCP `SYN-SENT` — the driver's SG had no
   inbound rule for the spare's cluster SG. (Worked around live by hand-adding an ingress rule for
   7777-7780 from the VPC CIDR to the driver SG.)
2. Even after the spare registered, the actual multi-stream recovery then failed with `TimeoutError` at
   `recovery.py:309` (`sstate["read"].recv()`): when the tail drained, the predecessor stage opened a
   **new** data connection to the spare's `:7777`, which the **spare's own** (separate-cluster) SG blocked
   inbound — so activations never reached the spare and the driver's read timed out, leaving the fleet
   degraded.

**Fix (designed, not yet re-validated live):** pin both `cairn-dsv4.sky.yaml` and `cairn-dsv4-spare.sky.yaml`
to one **shared named security group** via SkyPilot's `config.aws.security_group_name: cairn-fleet`, so every
box — live fleet and every replenished spare — joins the same SG and all of 7777-7780 are mutually reachable
with no per-launch SG surgery. A one-time-per-region helper (`infra/aws/ensure-fleet-sg.sh`) creates that SG
with the right rules (`:22` for provisioning + 7777-7780 across the VPC). A fragile per-launch fallback
(authorize the ports on both SGs after launch) is available opt-in via `CAIRN_AUTHORIZE_RECOVERY_PORTS=1` in
the watcher. **Lesson:** the recovery design implicitly assumed all boxes share one SG (true for a single
multi-node cluster); the moment a spare is provisioned as a *separate* cluster, that assumption silently
breaks — a reminder that the self-replenish path needs its own live network test, not just the in-fleet swap.

## The single most important caveat

**Spare warm-up time (~20–40 min to load the 294 GB checkpoint) is the current ceiling on
how aggressively the system can absorb reclaims.** One swap is invisible; a burst of swaps
faster than spares can warm would eventually exhaust the pool. The fix is operational
(snapshots / pre-baked images / larger standing pool / faster NVMe staging), not research —
but it's the thing that most separates "demo that recovers" from "service you trust."

## Recommended next steps (in order)

1. **Finish the live recovery matrix** (1–2 sessions) → the claims are fully backed.
2. **Run the head-to-head benchmark** → the headline number.
3. **One-command reproducible bring-up + operator README** → others can run it.
4. **Then** decide whether to deploy the control plane (Target B) — only if the goal moves
   beyond artifact/paper toward a service.

Bottom line: **the breakthrough is behind us; the remaining work is known, bounded, and
mostly operational.** For the stated open-source + paper goal, we're close. For a service
others lean on, the durability + warm-up + multi-tenancy work is the real distance.
