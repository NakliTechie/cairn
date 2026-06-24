# Cairn — S3-backed artifact cache

A durable plan for not paying for the same expensive setup twice. Born from the 2026-06-24
DeepSeek-V4-Flash maiden run, where each fresh g7e box paid a ~30-min CUDA build + a 150 GB
weight pull before serving a single token.

The strategy ties directly into **recovery and hotswaps**: a box that comes up fast is a box that
can replace a spot-reclaimed stage fast, promote a spare fast, or load a different model fast. Cache
= cheaper experiments *and* a better recovery story (Path 2 / durable-star).

## The constraint that shapes everything

**Boxes have no AWS credentials** (no instance role — verified 2026-06-24). So:
- **Restore (S3 → box) is automatic** via SkyPilot `file_mounts: { /path: s3://... }` — SkyPilot
  uses the *controller's* creds to sync. Zero box-side credentials.
- **Populate (box → S3) is a manual controller step** — the box can't write S3. Pattern:
  pull the artifact off a live box (`ssh box 'tar czf - X' > local.tgz`), then
  `aws s3 cp local.tgz s3://...` from the laptop/controller.

→ **Populate-once, restore-many.** This is fine: artifacts change rarely (only when their key changes).

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

### 2. Weights — NEXT (the bigger win, and the hotswap enabler)
- **Why in-region:** the boxes run in eu-south-2. A same-region S3 mirror means each box pulls 150 GB
  from S3 over the AWS backbone — faster than HF, no HF rate-limits when 4+ boxes pull at once, and no
  ~600 GB/run egress from HF. Cross-region defeats the purpose; **the weight bucket must live in the
  launch region** (unlike the 4.7 MB kernel, where cross-region is negligible).
- **Mechanism:** SkyPilot S3 file_mount (or `sky storage`) syncs `s3://cairn-weights-eu-south-2/
  deepseek-v4-flash/<revision>/` → `~/model`. Restore is automatic; populate once from a box that
  already pulled from HF (or a one-off controller-side `aws s3 sync`).
- **Cost:** ~150 GB × ~$0.023/GB-mo ≈ **$3.5/mo** per cached model. Trivial vs the GPU-time saved.
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

## Next steps
- [x] Kernel build → S3, wired into the dsv4 yaml (2026-06-24).
- [ ] Create `cairn-weights-<region>` bucket(s) **in the launch region**; populate DeepSeek-V4-Flash
      once; add a weights file_mount + setup branch (S3 hit → skip HF). Gate: after the maiden decode.
- [ ] Generalize: a small helper (`infra/skypilot/cache.py`?) — `populate <artifact>` /
      `key <artifact>` — so kernel + weights + future models share one keying/populate convention.
- [ ] ECR image mirror — only if Docker Hub pull time is measured as a problem.
- [ ] Tie weight-cache into the recovery path: spare/replacement boxes pull from in-region S3 →
      measure the MTTR improvement vs HF (feeds the Path 2 / headline recovery numbers).
