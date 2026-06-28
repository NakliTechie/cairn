# Cairn — Warm-AMI spares (single-digit-minute replenishment)

_Design only — 2026-06-28. Nothing here is deployed. Sibling of [`CACHING.md`](./CACHING.md):
caching makes the **first** box in a region cheap; the warm AMI makes a **replacement** box
fast._

## The problem this solves (and the one it doesn't)

Cairn keeps a warm standing spare so a spot reclaim is invisible — when a stage dies, the
driver promotes the already-loaded spare in **sub-second** time (the slice is already in VRAM,
flashinfer is already warmed). That swap latency is **not** the problem.

The problem is **replenishment**: once a spare is consumed by a promotion, the warm pool is
down one, and [`replenish-watcher.sh`](./replenish-watcher.sh) `sky launch`es a fresh
[`cairn-dsv4-spare.sky.yaml`](./cairn-dsv4-spare.sky.yaml) box to refill it. That fresh box
must, from cold:

1. provision a spot instance + boot the DLAMI,
2. pull the ~82 GB sglang Docker image (in-region ECR mirror, ~25 GB compressed),
3. **`aws s3 sync` the ~294 GB DeepSeek-V4-Flash FP8 checkpoint S3 → NVMe**,
4. start the container, load its slice into VRAM, flashinfer-warm, then `--announce`.

**Observed live 2026-06-28: ~30+ min, dominated by step 3** (the S3→NVMe weight sync). That is
[`report/productization.md`](../../report/productization.md)'s single biggest operational
ceiling: if reclaims arrive faster than ~30 min apart for sustained stretches, replenishment
can't keep up and the warm pool drains.

> **Scope.** This optimizes **replenishment latency** (how fast a consumed spare is replaced),
> *not* **swap latency** (already sub-second). It makes the pool refill in single-digit minutes
> so the system can absorb deeper, faster reclaim storms before the pool is exhausted.

## Why you can't just snapshot the running box

The obvious idea — "snapshot a warmed box, relaunch from it" — has one hard constraint:

> **The g7e NVMe at `/opt/dlami/nvme` is INSTANCE STORE (ephemeral local disk). It is NOT
> captured in an AMI. AMIs only snapshot EBS volumes.**

Today both the weights and the Docker image live on that NVMe (see step 0 of the spare yaml's
`setup:` — we moved them there for the multi-GB/s reshape-reload bandwidth and to drop
`disk_size` 600→150). An AMI of that box would capture the OS root and *nothing else* — the
294 GB would be gone on the next boot, and we'd re-sync from S3 anyway. **No win.**

To bake the weights into an image, they must sit on an **EBS** volume that the AMI's block
device mapping (BDM) references — either the root volume or, better, a dedicated **EBS data
volume** that the AMI restores-and-attaches automatically on launch.

## The design: DLAMI root + a baked EBS "warm-data" volume

```
Warm AMI  =  DLAMI root (R580 driver, sglang prereqs)   ← unchanged base
          +  EBS data volume from a snapshot, in the AMI's BDM:
                /opt/cairn-warm/model   ← 294 GB FP8 checkpoint (baked)
                /opt/cairn-warm/docker  ← docker data-root w/ sglang image pre-pulled (~82 GB)
```

On launch the instance's BDM auto-creates the data volume from its snapshot and attaches it; an
`/etc/fstab` entry baked into the AMI mounts it at `/opt/cairn-warm` on boot. So a box launched
from this AMI comes up with **both the image and the weights already on local EBS** — steps 2
and 3 above become no-ops.

### The lazy-load gotcha — and why Fast Snapshot Restore (FSR) is the linchpin

An EBS volume restored from a snapshot does **not** copy its blocks up front. It **lazy-loads
each block from S3 on first access.** So a naive warm AMI trades the explicit `aws s3 sync` for
an *implicit, slower, less-controllable* hydration: the first model-load touches all the slice's
blocks and each first-touch stalls on an S3 round-trip. First-load on a lazily-restored volume
is typically **slower** than a saturated parallel `s3 sync`, not faster. The image baking still
helps (no 82 GB pull), but the weight read — the dominant cost — is not fixed.

Two ways to defeat lazy-load:

- **(a) Fast Snapshot Restore (FSR)** — enable FSR on the snapshot, **per Availability Zone**.
  Volumes restored from an FSR-enabled snapshot in that AZ are **fully initialized with full
  provisioned performance immediately** — no lazy-load penalty, no first-touch stall. This is
  the linchpin. Enabling FSR on a snapshot is a **one-time ~up-to-60-min** operation per
  snapshot-per-AZ (state goes `enabling`→`optimizing`→`enabled`); after that, **every** volume
  restore in that AZ is instant. FSR is billed **$0.75/hr per snapshot per AZ while enabled**
  (cost section below).
- **(b) Pre-warm / provisioned read** — without FSR, you can force hydration by reading every
  block once on boot (`fio`/`dd` the device), or accept the lazy-load penalty on the first box
  in an AZ. This just moves the slow read to boot time; it's the fallback when FSR's standing
  cost isn't justified. (Provisioning higher IOPS/throughput on the *volume* does not remove the
  lazy-load penalty — that penalty is the S3 hydration, independent of volume type.)

**FSR is what turns the warm AMI from "saves the image pull" into "single-digit-minute
replenishment."**

### Reads still ride EBS, not NVMe — keep NVMe for reshape

With the weights on an FSR'd EBS volume, the spare loads its slice (~58 GB) **directly from EBS
into VRAM** — a one-time bounded read, ~1–2 min at gp3's 1 GB/s ceiling (or faster on io2 Block
Express, up to ~4 GB/s on this Nitro instance). That's fast enough for the announce critical
path.

The one regression vs today: NVMe gave multi-GB/s for **load-on-promotion reshape** (a generic
spare re-reads a *different* slice from disk when it's promoted to a non-tail rank — the
`--stage-partition` path). EBS gp3 caps lower. Mitigation, **off the critical path**: after the
spare announces, a background job copies `/opt/cairn-warm/model` → `/opt/dlami/nvme/model`
(~294 GB at ~1 GB/s ≈ 5 min). The announce doesn't wait on it; once it lands, later reshapes
read NVMe at full bandwidth. (Or provision the data volume as **io2 Block Express** and skip the
copy — simpler, slightly pricier per-GB-month.)

## Warm-up breakdown: now vs AMI vs AMI+FSR

g7e.2xlarge, in-region. "Slice" = one stage's ~58 GB resident shard of the 294 GB checkpoint.

| Phase | Now (S3 sync) | Warm AMI, no FSR | **Warm AMI + FSR** |
|---|---|---|---|
| `sky launch` → spot instance running + DLAMI boot | ~3–5 min | ~3–5 min | ~3–5 min |
| Mount baked data volume | — | <1 min | <1 min |
| Docker image acquire (ECR ~25 GB pull + extract) | ~3–6 min¹ | **0 (baked)** | **0 (baked)** |
| **Weight availability (294 GB)** | **~20–30 min** (S3→NVMe sync) | ~8–15 min (lazy-load on first read) | **~0 (pre-initialized)** |
| Kernel restore (cached `.so`) | <1 min | <1 min | <1 min |
| Container start + slice→VRAM (~58 GB) + flashinfer warm | ~3–5 min | ~3–5 min² | ~2–4 min |
| **Total to `--announce`** | **~30–40 min** | **~12–20 min** | **~5–8 min** |

¹ Concurrent with the weight sync today, so it hides under step 3's wall-clock.
² Inflated because the slice load *is* the lazy-load in the no-FSR case.

With FSR, the floor is set by the irreducible costs — **spot acquisition + boot + flashinfer
warm** — not by moving 294 GB. That's the whole point: the data no longer moves at replenish
time; it was baked once at build time.

## Build procedure

Rebuild is **occasional, not constant** — only when the **model** or the **sglang image**
changes (model-as-config). Between rebuilds the AMI is reused for every replenish. The build is
scripted in [`build-warm-ami.sh`](./build-warm-ami.sh) (written, **not run** — it provisions and
snapshots real resources and uses admin-tier EC2 perms). Shape:

1. **Launch a builder box** from the region's DLAMI with an extra blank ~400 GB gp3 data volume
   attached (a g7e isn't required for staging, but using the DLAMI base keeps the resulting
   AMI's root identical to production).
2. **Stage onto the data volume** mounted at `/opt/cairn-warm`:
   - `docker` data-root → `/opt/cairn-warm/docker`; pull the sglang image from the in-region ECR
     mirror (same path as `acquire_image()` in the spare yaml).
   - `aws s3 sync s3://skypilot-cairn-weights-<region>/deepseek-v4-flash-fp8/main` →
     `/opt/cairn-warm/model` (same source as `acquire_weights()`).
   - restore the 0xSero kernel `.so` too (optional; it's tiny and already S3-cached).
   - add the `/opt/cairn-warm` mount to `/etc/fstab` so it auto-mounts on boot.
3. **Stop the instance** (clean unmount → consistent snapshot).
4. **`create-image`** off the stopped builder → one AMI whose BDM captures DLAMI-root + the
   warm-data volume's snapshot. (Keeping every artifact on the *data* volume means root stays a
   pristine DLAMI.)
5. **Replicate per region** with `copy-image` to each actively-used region (AMIs are
   region-scoped). One-time cross-region data-transfer cost; storage then accrues per region.
6. **(At run time, not build time) enable FSR** on the data-volume snapshot for the AZ(s) the
   fleet will launch into. Allow up to ~60 min to reach `enabled`.
7. **Output: the per-region AMI id** → drop into the spare yaml's `image_id` (and the main yaml).

The script keeps build-time EC2 mutations (`run-instances`, `create-image`, `copy-image`, FSR
enable/disable) behind `AWS_PROFILE=${AWS_PROFILE:-admin-cli}` — the least-priv `cairn-skypilot`
key intentionally **cannot** do these (consistent with the
credential posture (scoped least-privilege key; see `infra/aws/`)). It prints the resulting AMI id and never tears anything
down on its own.

## Integrating with the replenish path

Today the watcher launches the spare yaml with `--region <R> --image-id <DLAMI>`. With a warm
AMI, the **only structural change** is the image and a couple of setup skips:

1. **`image_id` → the region's warm AMI** (per region, like the DLAMI ids already tracked in
   `cairn-dsv4.sky.yaml`). The watcher already forwards `CAIRN_IMAGE_ID` from the live box's
   IMDS `ami-id`; point the fleet at the warm AMI and replacements inherit it automatically.
2. **`setup:` becomes detect-and-skip.** Guard `acquire_image` / `acquire_weights` on the baked
   artifacts so a warm-AMI box no-ops them and a plain-DLAMI box still works (graceful
   degradation — the yaml stays launchable without the warm AMI):

   ```sh
   WARM=/opt/cairn-warm
   if [ -d "$WARM/model" ] && [ -f "$WARM/model/config.json" ]; then
     MODEL_DIR="$WARM/model"; echo "[setup] warm-AMI: weights baked at $MODEL_DIR — skip S3 sync"
   else
     # ... existing NVMe staging + acquire_weights path ...
   fi
   if [ -n "$WARM_DOCKER" ] && sudo docker image inspect "$IMG" >/dev/null 2>&1; then
     echo "[setup] warm-AMI: image baked — skip pull"
   else
     acquire_image
   fi
   # Optional, OFF the announce critical path: background copy weights EBS→NVMe for fast reshape.
   ```

3. **The `run:` block is unchanged in spirit** — it still points `--model` at `MODEL_DIR`,
   loads slice `CAIRN_SPARE_RANK` into VRAM, flashinfer-pre-warms, dials
   `$DRIVER_HOST:$DRIVER_SPARE_SINK`, and `--announce`s its routable IP. **A warm-AMI spare is
   still a real warmed spare** — it does the full VRAM load + flashinfer warm before announcing,
   so the driver only ever promotes a genuinely-ready node. The AMI removes the *staging* cost
   (image pull + weight movement), not the *warming* step.

This is deliberately additive: a fleet launched on the DLAMI keeps working; a fleet launched on
the warm AMI replenishes ~5× faster. The actual yaml edits are left as a follow-up (kept out of
the live, working spare yaml until the AMI exists) — the snippet above is the exact diff.

## Per-region / per-AZ + cost tradeoffs

| Item | Cost | Notes |
|---|---|---|
| AMI registration | $0 | The AMI is metadata; you pay for its backing snapshot. |
| Snapshot storage | **~$15–20/mo per region** | ~376 GB (294 weights + 82 image) × ~$0.05/GB-mo. FP8 + image layers are high-entropy → ~no snapshot compression. Lazy/incremental: a rebuild that only changes weights re-stores only changed blocks. |
| Cross-region copy | ~$0.02/GB one-time per region | `copy-image`; then per-region storage as above. |
| **FSR (the swing cost)** | **$0.75/hr per snapshot per AZ** | Always-on ≈ **$540/mo per AZ**. Only worth always-on for a 24/7 endpoint where single-digit replenish is load-bearing. |

**FSR is the cost decision.** Two regimes:

- **Session / soak runs (the common case):** enable FSR on the snapshot for the run's AZ at
  fleet bring-up (it reaches `enabled` within ~60 min — during that first hour, replenishment
  falls back to lazy-load or the S3-sync path), keep it on for the run, **disable at teardown**.
  A 6 h session in one AZ ≈ **$4.50**. Negligible.
- **Always-on endpoint:** $540/mo/AZ buys guaranteed single-digit replenishment. Trim it by
  pinning the fleet (and FSR) to **one AZ per region** rather than spreading. Weigh against the
  alternatives below.

**AZ pinning matters.** FSR is per-AZ and EBS volumes are AZ-local. Pin the fleet's spares to
the AZ(s) where FSR is enabled, or a spare landing in a non-FSR AZ silently falls back to
lazy-load. SkyPilot spot placement should be constrained to the FSR AZ(s) for the region.

Snapshot storage is cheap and lazy-per-region (mirror the CACHING.md pattern: build/copy the AMI
only into regions you actually launch in). FSR is the only meaningful recurring cost, and it's
opt-in per run.

## Alternatives considered

- **Warm EBS volume pool (no snapshot/FSR).** Pre-create N persistent gp3/io2 volumes per AZ,
  each holding the baked weights, and `attach-volume` one to each replacement on replenish. Avoids
  both lazy-load *and* FSR's hourly cost — you pay only flat volume-storage. But it's
  operationally heavier: managing a volume pool, AZ pinning, attach orchestration *outside*
  SkyPilot's lifecycle, and re-detach/re-attach races on spot churn. Worth it only if FSR's
  standing cost becomes the dominant line item on an always-on endpoint.
- **Deeper standing warm pool.** Orthogonal, not a substitute: more standing spares give more
  headroom *between* replenishments but each idle spare burns a full GPU-hour. Warm AMI makes
  each replenishment cheap; a deeper pool makes the system tolerant of slow ones. Best combined —
  a deeper pool *and* fast replenish.
- **Bigger/striped EBS for raw read speed.** Doesn't address lazy-load (the actual bottleneck);
  only matters once FSR has removed the hydration penalty, at which point gp3-1GB/s is already
  adequate for the one-time slice load.

## Risks / open questions

- **SkyPilot + custom BDM.** Confirm SkyPilot 0.12 launches from an AMI **without stripping the
  non-root data volume** in its BDM (it manages `disk_size` for root; the extra volume should
  ride the AMI's mapping). Verify on the first build that the data volume actually attaches.
- **FSR enable latency window.** The ~60-min `enabling`→`enabled` ramp means the *first* hour
  after enabling FSR has no fast-restore guarantee. For session runs, enable at (or just before)
  bring-up and accept lazy-load fallback for that first window; for always-on, FSR is already
  warm.
- **Rebuild discipline.** The AMI is now a build artifact keyed by (model revision, image
  digest, kernel ABI) — exactly the CACHING.md keys. A stale AMI silently serves an old
  model/image. Tie the rebuild trigger to the same digest bump that invalidates the kernel cache.
- **EBS vs NVMe reshape throughput** (the open question already in `pending.md`): the background
  EBS→NVMe copy resolves it off the critical path, but measure whether direct io2-BX reads are
  fast enough to skip the copy entirely.

## Bottom line

Bake the 294 GB checkpoint and the 82 GB image onto an **EBS data volume** in a per-region warm
AMI; enable **FSR** on its snapshot in the fleet's AZ so restored volumes are instantly
full-performance. Replenishment drops from **~30–40 min to ~5–8 min** — the 294 GB stops moving
at replenish time because it was baked once at build time. The spare still does a real VRAM load
+ flashinfer warm before announcing, so the swap stays sub-second and invisible. Cost is ~$15–20/mo
storage per region plus FSR ($/hr per AZ, opt-in per run) — small against turning the system's
single biggest operational ceiling into a non-issue.
</content>
</invoke>
