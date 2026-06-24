# Cairn — S3-backed artifact cache

A durable plan for not paying for the same expensive setup twice. Born from the 2026-06-24
DeepSeek-V4-Flash maiden run, where each fresh g7e box paid a ~30-min CUDA build + a 150 GB
weight pull before serving a single token.

The strategy ties directly into **recovery and hotswaps**: a box that comes up fast is a box that
can replace a spot-reclaimed stage fast, promote a spare fast, or load a different model fast. Cache
= cheaper experiments *and* a better recovery story (Path 2 / durable-star).

## The core principle: download once, fan out from in-region S3

**Every artifact is pulled from its slow public origin (HF, Docker Hub, a 30-min compile) EXACTLY
ONCE, into an in-region S3 bucket. Boxes only ever pull from S3.** This is the default, not an
optimization to add later.

Why it's not optional at scale — the weights make it obvious:

| Boxes | Per-box-pulls-HF (today) | Download-once-to-S3 (the rule) |
|---|---|---|
| 4 | 600 GB HF egress, 4× rate-limit exposure | 150 GB HF once + 4× in-region S3 (backbone) |
| 20 | **3 TB HF egress**, near-certain throttling, slow | 150 GB HF once + 20× in-region S3 (parallel, fast) |

S3→EC2 in the **same region** is the AWS backbone: high-throughput, parallel across boxes (S3 scales
horizontally), no public-internet egress, no third-party rate limits. HF (or Docker Hub) sees one
pull per artifact revision, ever. This is the network-friendly *and* the scalable choice — the same
decision, twice over.

**Corollary — boxes need read creds to the cache bucket.** The 5 MB kernel rides SkyPilot file_mounts
(controller-mediated). 150 GB × N boxes must NOT funnel through the controller — each box pulls S3
directly and in parallel, which means each box needs read access. The scalable answer is an **IAM
instance profile** scoped read-only to the cache bucket(s), attached to the fleet → boxes get
short-lived creds from the metadata service, no long-lived keys on disk, `aws s3 sync` runs directly.
(See "boxes have no creds" below — for scale, we *fix* that with a least-privilege read role, rather
than route everything through the controller.)

## The constraint that shapes everything

**Boxes have no AWS credentials** (no instance role — verified 2026-06-24). So:
- **Restore (S3 → box) is automatic** via SkyPilot `file_mounts: { /path: s3://... }` — SkyPilot
  uses the *controller's* creds to sync. Zero box-side credentials.
- **Populate (box → S3) is a manual controller step** — the box can't write S3. Pattern:
  pull the artifact off a live box (`ssh box 'tar czf - X' > local.tgz`), then
  `aws s3 cp local.tgz s3://...` from the laptop/controller.

→ **Populate-once, restore-many.** This is fine: artifacts change rarely (only when their key changes).

**For scale, fix the no-creds default with a read role.** The controller-mediated file_mount is fine
for the 5 MB kernel but is the wrong tool for 150 GB × N boxes (controller bottleneck). Attach a
**least-privilege IAM instance profile** to the fleet, scoped read-only to the cache bucket(s)
(`s3:GetObject`/`ListBucket` on `skypilot-cairn-weights-*` + `skypilot-cairn-artifacts`). Boxes then pull S3
directly and in parallel via metadata-service creds — no long-lived keys, no controller in the path.
This is additive to the existing least-priv posture: read-only, cache-bucket-only, no mutate.

Bucket: **`skypilot-cairn-artifacts`** (the `skypilot-*` name matches the cairn-skypilot IAM S3 scope,
so the least-priv key can read/write it; `s3:CreateBucket` on that pattern is allowed).

## The three artifacts — different decisions

| Artifact | Size | Origin | Decision | Key |
|---|---|---|---|---|
| **0xSero kernel build** | 15 MB | ~30-min CUDA compile on g7e | ✅ **CACHE** (done) | base-image **digest** + arch (sm_120) |
| **V4-Flash weights** | 150 GB | HF download (public) | ✅ **CACHE in-region** (planned) | model id + revision |
| **sglang base image** | 82 GB | `docker pull` (digest-pinned) | ❌ **don't S3** — ECR mirror if pulls hurt | image digest |

### 1. Kernel build — DONE (2026-06-24)
- Stored: `s3://skypilot-cairn-artifacts/dsv4-kernel/<image-digest>/build-docker.tar.gz`
  (current digest `408846afd1b0`).
- Wired: `cairn-dsv4.sky.yaml` file_mounts it to `/cairn-kernel-cache`; setup restores it (cheapest-
  first: local .so → S3 cache → build).
- **ABI caveat:** the `.so` is compiled against the base image's CUDA 12.9.1 + cpython-3.12 for
  sm_120. **If the image tag is re-pushed (digest changes) or we move off sm_120, the cache is stale**
  — update the digest in the file_mount and re-populate. A stale/missing prefix degrades gracefully
  to a full rebuild.

### 2. Weights — WIRED (lazy, 2026-06-24); populate is the remaining manual step
This is where "download once, fan out" pays off most. See the core principle above.
- **Buckets EXIST in all 4 launchable regions** (`skypilot-cairn-weights-{eu-south-2,us-east-2,
  us-west-2,ap-northeast-2}`) so any region can be the cheapest at launch time. The `skypilot-`
  prefix is REQUIRED (the cairn-skypilot IAM key's S3 scope is `skypilot-*`/`sky-*`). Cross-region
  defeats the purpose — `cairn-dsv4.sky.yaml` setup auto-detects the box's region (IMDSv2) and pulls
  from THAT region's bucket.
- **Fan out — WIRED:** setup step 4 does `aws s3 sync s3://skypilot-cairn-weights-<region>/
  deepseek-v4-flash/main ~/model` if the object exists, else falls back to HF. Each box pulls
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
      python3 -c \"from huggingface_hub import snapshot_download; snapshot_download('"'"'deepseek-ai/DeepSeek-V4-Flash'"'"', local_dir='"'"'/tmp/m'"'"', max_workers=8)\"
      aws s3 sync /tmp/m s3://$B/deepseek-v4-flash/main --region $R --only-show-errors"'
  # NOTE: the populate box needs S3 WRITE — pass cairn-skypilot creds (it has s3:* on skypilot-*), or
  # better, mint a scoped write key. cairn-s3-cache is read-only by design (it rides on GPU boxes).
  ```
- **Cost:** ~150 GB × ~$0.023/GB-mo ≈ **$3.5/mo** per cached model+region. Lazy ⇒ pay only for
  regions actually used. Drops to near-zero per *box* added (the win compounds with fleet size).
- **Hotswap / recovery payoff (the reason this matters beyond cost):**
  - **Spare promotion / post-reclaim replacement:** a replacement box loads weights from in-region S3
    in a fraction of the HF time → lower MTTR for the catastrophic-recovery path (Path 2 durable-star).
  - **Model hotswap:** keep several models pre-staged in S3; bringing up a fleet for a different model
    becomes "sync from S3 + load", not "re-download from HF". Enables fast A/B and multi-model packing.
  - This is the concrete infra under the "self-replenishing spare" + "resumable driver" items in
    `plan/pending.md` (Parked / Path 2).

### 3. Base image — leave on Docker Hub
It's a *pull*, not a build, and it's digest-pinned (`lmsysorg/sglang:deepseek-v4-blackwell@sha256:
408846…`). S3 wouldn't pull faster. **If** Docker Hub pulls become slow or rate-limited across many
boxes, the right fix is an **ECR mirror in the launch region** (`docker tag` + push once, pull from
ECR after) — not S3. Low priority until pulls are a measured bottleneck.

## Next steps (priority order — weights fan-out is the headline)
- [x] Kernel build → S3, wired into the dsv4 yaml (2026-06-24).
- [ ] **Read instance profile** for the fleet — least-priv `s3:GetObject`/`ListBucket` on
      `skypilot-cairn-weights-*` + `skypilot-cairn-artifacts`, attached at launch (confirm how SkyPilot 0.12
      attaches an IAM instance profile). This is the prerequisite for direct parallel S3 fan-out.
- [ ] **Weights download-once → in-region S3, fan-out to boxes.** Create `skypilot-cairn-weights-<region>`
      **in the launch region**; populate DeepSeek-V4-Flash once (HF→S3); change the dsv4 setup so each
      box does `aws s3 sync s3://skypilot-cairn-weights-<region>/<model>/<rev>/ ~/model` (S3 hit → skip HF),
      falling back to HF only on a cache miss. Gate: right after the maiden decode. This is THE
      network-friendly pattern and the prerequisite for 10-20-box runs.
- [ ] Generalize: a small helper (`infra/skypilot/cache.py`?) — `populate <artifact>` /
      `key <artifact>` / `sync <artifact>` — so kernel + weights + future models share one
      keying/populate/restore convention (and the same instance-profile read path).
- [ ] Tie weight-cache into the recovery path: spare/replacement boxes pull from in-region S3 →
      measure the MTTR improvement vs HF (feeds the Path 2 / headline recovery numbers).
- [ ] ECR image mirror — only if Docker Hub pull time is measured as a problem (lowest priority).

> **Scale note (the reason this is the default, not an optimization):** at 10-20 boxes, per-box HF
> pulls = multi-TB egress per launch + rate-limit throttling + slow bring-up. Download-once-to-S3 +
> in-region parallel fan-out keeps HF egress flat (1× per model revision) regardless of fleet size,
> and makes each added box nearly free to provision. It's the only pattern that scales.
