# Cairn — S3-backed artifact cache

A durable plan for not paying for the same expensive setup twice. Without it, each fresh g7e box
pays a ~30-min CUDA build + a 294 GB weight pull before serving a single token.

The strategy ties directly into **recovery and hotswaps**: a box that comes up fast is a box that
can replace a spot-reclaimed stage fast, promote a spare fast, or load a different model fast. Cache
= cheaper experiments *and* a better recovery story.

## The core principle: download once, fan out from in-region S3

**Every artifact is pulled from its slow public origin (HF, Docker Hub, a 30-min compile) EXACTLY
ONCE, into an in-region S3 bucket. Boxes only ever pull from S3.** This is the default, not an
optimization to add later.

Why it's not optional at scale — the weights make it obvious:

| Boxes | Per-box-pulls-HF (naive) | Download-once-to-S3 (the rule) |
|---|---|---|
| 4 | 600 GB HF egress, 4× rate-limit exposure | 294 GB HF once + 4× in-region S3 (backbone) |
| 20 | **3 TB HF egress**, near-certain throttling, slow | 294 GB HF once + 20× in-region S3 (parallel, fast) |

S3→EC2 in the **same region** is the AWS backbone: high-throughput, parallel across boxes (S3 scales
horizontally), no public-internet egress, no third-party rate limits. HF (or Docker Hub) sees one
pull per artifact revision, ever. This is the network-friendly *and* the scalable choice — the same
decision, twice over.

**Corollary — boxes need read creds to the cache bucket.** The 5 MB kernel rides SkyPilot file_mounts
(controller-mediated). 294 GB × N boxes must NOT funnel through the controller — each box pulls S3
directly and in parallel, which means each box needs read access. The scalable answer is an **IAM
instance profile** scoped read-only to the cache bucket(s), attached to the fleet → boxes get
short-lived creds from the metadata service, no long-lived keys on disk, `aws s3 sync` runs directly.
(See "boxes have no creds" below — for scale, we *fix* that with a least-privilege read role, rather
than route everything through the controller.)

## The constraint that shapes everything

**Boxes have no AWS credentials** (no instance role by default). So:
- **Restore (S3 → box) is automatic** via SkyPilot `file_mounts: { /path: s3://... }` — SkyPilot
  uses the *controller's* creds to sync. Zero box-side credentials.
- **Populate (box → S3) is a manual controller step** — the box can't write S3. Pattern:
  pull the artifact off a live box (`ssh box 'tar czf - X' > local.tgz`), then
  `aws s3 cp local.tgz s3://...` from the laptop/controller.

→ **Populate-once, restore-many.** This is fine: artifacts change rarely (only when their key changes).

**For scale, fix the no-creds default with a read role.** The controller-mediated file_mount is fine
for the 5 MB kernel but is the wrong tool for 294 GB × N boxes (controller bottleneck). Attach a
**least-privilege IAM instance profile** to the fleet, scoped read-only to the cache bucket(s)
(`s3:GetObject`/`ListBucket` on `skypilot-cairn-weights-*` + `skypilot-cairn-artifacts`). Boxes then pull S3
directly and in parallel via metadata-service creds — no long-lived keys, no controller in the path.
This is additive to the existing least-priv posture: read-only, cache-bucket-only, no mutate.

Bucket: **`skypilot-cairn-artifacts`** (the `skypilot-*` name matches the cairn-skypilot IAM S3 scope,
so the least-priv key can read/write it; `s3:CreateBucket` on that pattern is allowed).

## The three artifacts — different decisions

| Artifact | Size | Origin | Decision | Key |
|---|---|---|---|---|
| **0xSero kernel build** | 15 MB | ~30-min CUDA compile on g7e | ✅ **CACHE** | base-image **digest** + arch (sm_120) |
| **V4-Flash weights** | 294 GB | HF download (public) | ✅ **CACHE in-region** | model id + revision |
| **sglang base image** | 82 GB | `docker pull` (digest-pinned) | ✅ **in-region ECR mirror** (lazy, per region) | image digest |

### 1. Kernel build
- Stored: `s3://skypilot-cairn-artifacts/dsv4-kernel/<image-digest>/build-docker.tar.gz`
  (current digest `408846afd1b0`).
- Wired: `cairn-dsv4.sky.yaml` file_mounts it to `/cairn-kernel-cache`; setup restores it (cheapest-
  first: local .so → S3 cache → build).
- **ABI caveat:** the `.so` is compiled against the base image's CUDA 12.9.1 + cpython-3.12 for
  sm_120. **If the image tag is re-pushed (digest changes) or we move off sm_120, the cache is stale**
  — update the digest in the file_mount and re-populate. A stale/missing prefix degrades gracefully
  to a full rebuild.

### 2. Weights — wired (lazy); populate is a manual step
This is where "download once, fan out" pays off most. See the core principle above.
- **Buckets EXIST in all 4 launchable regions** (`skypilot-cairn-weights-{eu-south-2,us-east-2,
  us-west-2,ap-northeast-2}`) so any region can be the cheapest at launch time. The `skypilot-`
  prefix is REQUIRED (the cairn-skypilot IAM key's S3 scope is `skypilot-*`/`sky-*`). Cross-region
  defeats the purpose — `cairn-dsv4.sky.yaml` setup auto-detects the box's region (IMDSv2) and pulls
  from THAT region's bucket.
- **Fan out — WIRED:** setup step 4 does `aws s3 sync s3://skypilot-cairn-weights-<region>/
  deepseek-v4-flash-fp8/main ~/model` if the object exists, else falls back to HF. Each box pulls
  directly + in-region (not via the controller). Auth: the **read-only `cairn-s3-cache` key**
  (CAIRN_S3_CACHE_* in secrets.env, passed via --env) — chosen over an instance profile for now
  (simpler; instance profile is the scale follow-up below).
- **Populate (lazy, per region+model) — MANUAL, the remaining step.** First launch in a region is a
  cache MISS (HF fallback). To make the NEXT launch there a hit, stage the bucket once via a cheap
  in-region CPU box (HF → S3), using cairn-skypilot WRITE creds (transient box, torn down):
  ```sh
  # one-time per region R (e.g. R=us-east-2). ~$0.10, ~30 min, no GPU.
  sky launch -c wpop --cloud aws --region $R --instance-type m6i.2xlarge --disk-size 400 -y --down \
    --env HF_TOKEN --env R=$R --env B=skypilot-cairn-weights-$R 'bash -c "
      export HF_HUB_ENABLE_HF_TRANSFER=1
      pip install -q hf_transfer huggingface_hub awscli
      python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('"'"'sgl-project/DeepSeek-V4-Flash-FP8'"'"', local_dir='"'"'/tmp/m'"'"', max_workers=8)\"
      aws s3 sync /tmp/m s3://$B/deepseek-v4-flash-fp8/main --region $R --only-show-errors"'
  # NOTE: the populate box needs S3 WRITE — pass cairn-skypilot creds (it has s3:* on skypilot-*), or
  # better, mint a scoped write key. cairn-s3-cache is read-only by design (it rides on GPU boxes).
  ```
- **Cost:** ~294 GB × ~$0.023/GB-mo ≈ **$3.5/mo** per cached model+region. Lazy ⇒ pay only for
  regions actually used. Drops to near-zero per *box* added (the win compounds with fleet size).
- **Hotswap / recovery payoff (the reason this matters beyond cost):**
  - **Spare promotion / post-reclaim replacement:** a replacement box loads weights from in-region S3
    in a fraction of the HF time → lower MTTR for the catastrophic-recovery path.
  - **Model hotswap:** keep several models pre-staged in S3; bringing up a fleet for a different model
    becomes "sync from S3 + load", not "re-download from HF". Enables fast A/B and multi-model packing.
  - This is the concrete infra under the self-replenishing spare + resumable driver.

### 3. Base image — in-region ECR mirror (lazy, per region)
It's a *pull*, not a build, and it's digest-pinned (`lmsysorg/sglang:deepseek-v4-blackwell@sha256:
408846…`). S3 wouldn't pull faster, so it stays on Docker Hub as the **fallback**; the **in-region ECR
mirror** is the fast path (compressed registry content ~25.6 GB, AWS-backbone, layer-cached).

**Region-derived, NOT hardcoded.** The dsv4 setup (`acquire_image()` in both `cairn-dsv4.sky.yaml`
and `cairn-dsv4-spare.sky.yaml`) derives `ECR_REGION` from IMDS (`placement/region`) — exactly like
`acquire_weights()` — so the mirror pull is always in-region. A hardcoded region would make a fleet
(or a replenished spare) launched elsewhere pull the image **cross-region** — slow, every box, every
spare. If the in-region pull misses (repo not yet populated there), it falls back to the ~82 GB
Docker Hub pull.

**Lazy-per-region, like the weights cache.** ECR repos are region-scoped, so the `cairn-sglang` mirror
must be populated **once per actively-used region**. Mirror from a live box that already pulled the
image, overriding `ECR_REGION` to the cluster's region (the helper ensure-creates the repo):
```sh
set -a; source infra/secrets.env; set +a
CLUSTER=cairn-dsv4 ECR_REGION=us-east-2 bash infra/skypilot/cache-image.sh   # Ohio; repeat per region used
```
Creating a *new* region's repo needs `ecr:CreateRepository` (the `cairn-s3-populate` key has R/W on the
existing repo only); the script tries the `default` profile (keyless root session; `admin-cli` was deleted 2026-08-01), else prints the one-line create command.
First launch in an un-mirrored region is a Docker Hub fallback (correct, just slower) — populate after so
the next launch / spare there is an in-region hit.

## Next steps (priority order — weights fan-out is the headline)
- [x] Kernel build → S3, wired into the dsv4 yaml.
- [ ] **Read instance profile** for the fleet — least-priv `s3:GetObject`/`ListBucket` on
      `skypilot-cairn-weights-*` + `skypilot-cairn-artifacts`, attached at launch (confirm how the
      pinned SkyPilot version attaches an IAM instance profile). Prerequisite for direct parallel S3 fan-out.
- [ ] **Weights download-once → in-region S3, fan-out to boxes.** Create `skypilot-cairn-weights-<region>`
      **in the launch region**; populate DeepSeek-V4-Flash once (HF→S3); change the dsv4 setup so each
      box does `aws s3 sync s3://skypilot-cairn-weights-<region>/<model>/<rev>/ ~/model` (S3 hit → skip HF),
      falling back to HF only on a cache miss. This is THE network-friendly pattern and the prerequisite
      for 10-20-box runs.
- [ ] Generalize: a small helper (`infra/skypilot/cache.py`?) — `populate <artifact>` /
      `key <artifact>` / `sync <artifact>` — so kernel + weights + future models share one
      keying/populate/restore convention (and the same instance-profile read path).
- [ ] Tie weight-cache into the recovery path: spare/replacement boxes pull from in-region S3 →
      measure the MTTR improvement vs HF.
- [x] ECR image mirror — region-derived (IMDS) in `acquire_image()`, lazy-per-region populate via
      `cache-image.sh`. Follow-up: a standing `ecr:CreateRepository` perm so new-region repos
      auto-create without the `default` profile.

> **Scale note (the reason this is the default, not an optimization):** at 10-20 boxes, per-box HF
> pulls = multi-TB egress per launch + rate-limit throttling + slow bring-up. Download-once-to-S3 +
> in-region parallel fan-out keeps HF egress flat (1× per model revision) regardless of fleet size,
> and makes each added box nearly free to provision. It's the only pattern that scales.
