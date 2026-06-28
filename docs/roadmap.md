# Cairn — roadmap & open directions

Planned optimizations and areas worth experimenting with. For the honest "how far from a
product" gap analysis see [`../report/productization.md`](../report/productization.md); for
the systems write-up see [`paper-draft.md`](paper-draft.md).

## Planned optimizations

### Warm-AMI spares + Fast Snapshot Restore (the warm-up / replenish ceiling)

The hot-swap itself is already sub-second — a reclaim is handled by stitching the dead slice
onto an **already-warm standing spare**. The only slow part is **refilling the pool
afterward**: today a replacement spare network-copies the ~294 GB checkpoint from S3 to its
NVMe, which takes **~20–40 min**. That replenish latency is the one thing that could let a
*burst* of reclaims outrun the pool.

Two levers, both about how fast the pool re-arms:

1. **Warm AMI.** Bake the weights + the container image into a per-region AMI (on an EBS
   volume — the NVMe instance store is ephemeral and can't be baked). A fresh spare then
   boots with everything already attached: **~5–10 min** to ready (lazy EBS hydration), no
   294 GB network copy and no image pull.
2. **+ Fast Snapshot Restore (FSR).** FSR pre-initializes the restored EBS volume, so there's
   no first-read hydration from S3. Warm-up collapses to just *instance boot + load the slice
   into VRAM + kernel warm* — **theoretically ~3–5 min**, bounded only by spot-provision and
   model-load time. (FSR is a one-time enable, up to ~60 min per snapshot per AZ; every
   restore after that is instant.)

**The key consequence — fast *next-spare-after-hotswap*.** When a hotswap consumes the warm
spare, the pool is momentarily thin. With FSR, the *replacement* spare comes up in single-digit
minutes, so the pool re-arms well before another reclaim is statistically likely — the fleet
stays continuously protected instead of having a ~30 min vulnerability window after each swap.
This is the difference between "recovers once" and "recovers repeatedly under sustained churn."

_Status: designed, not deployed (see the warm-AMI design doc + task). Not needed until
replenish latency becomes the binding constraint; the standing pool already makes the swap
itself instant._

## Further areas to experiment

- **Deeper standing warm pool (`warm_target ≥ 2`) vs cost.** A larger pool absorbs bursty or
  correlated reclaims (e.g. a whole AZ losing capacity) at the cost of idle GPUs. Find the
  sweet spot per workload; pairs with FSR replenishment above.
- **Push past the ~25 tok/s throughput ceiling.** The current fleet saturates around 4
  concurrent streams. Worth trying: speculative decoding, fatter layer slices (fewer hops →
  less per-token pipeline latency), overlapping wire-transfer with compute, larger
  micro-batches.
- **Multi-AZ / multi-region spot hunting.** Spread the fleet to lower correlated-reclaim risk
  and chase the cheapest capacity; today it's single-AZ. Prices and availability move (the
  launch sweep already exists; make placement multi-AZ-aware).
- **Interior reactive-death detection (heartbeats, "Path 2").** An abrupt (non-warned) kill
  of a *middle* node currently can't be uniquely identified without heartbeats — only the tail
  is cleanly detectable reactively. Proactive drain covers it today; heartbeats would close
  the reactive interior case.
- **Catastrophic durability (Cloudflare DO + resumable driver).** Mirror the durable token
  history off-box so the fleet survives losing *all* GPUs or the driver box — a durable-STAR,
  not a peer mesh (designed, sim-tested, not deployed).
- **Model-as-config breadth.** Pack and serve other models (GLM-5.2, Llama-70B FP8) on the
  same machinery; experiment with NVFP4 experts for more VRAM headroom → fewer boxes per model
  or a bigger model per box.
- **Autoscaling.** Scale N (pipeline depth) and stream capacity with demand rather than a
  fixed topology.
- **Head-to-head benchmarks.** Validate the headline against SpotServe / Petals / KevlarFlow
  on their own models — protocol + comparison table in
  [`../report/benchmark-plan.md`](../report/benchmark-plan.md).
