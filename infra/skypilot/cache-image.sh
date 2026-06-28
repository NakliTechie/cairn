#!/usr/bin/env bash
# Mirror the Blackwell sglang image into the IN-REGION ECR cache from a LIVE cluster box that already
# pulled it (so we never re-pull 82 GB from Docker Hub on future launches). The dsv4 YAML setup prefers
# this mirror (see "2. Acquire the official Blackwell V4 sglang image"); Docker Hub is the fallback.
#
# WHY ECR not docker-save-to-S3: `docker save` writes UNCOMPRESSED layers (~82 GB on disk); ECR stores
# the COMPRESSED registry content (~25 GB) and pulls are layer-cached + in-region — strictly better
# (the 2026-06-25 "image-cache marginal" finding was about the S3-tarball path; ECR is the real fix).
#
# Auth: the scoped cache key cairn-s3-populate (CAIRN_S3_WRITE_* in infra/secrets.env) — ECR R/W on the
# cairn-sglang repo only. Run AFTER a launch's image pull, BEFORE teardown.
#   set -a; source infra/secrets.env; set +a
#   CLUSTER=cairn-dsv4 bash infra/skypilot/cache-image.sh
#
# PER-REGION (lazy, like the weights cache): ECR repos are region-scoped, and the dsv4 setup now derives
# its ECR_REGION from IMDS (placement/region) — so the mirror MUST exist in whatever region the fleet
# launches in, else the box's in-region pull misses and falls back to a slow ~82 GB Docker Hub pull. Mirror
# once per actively-used region by overriding ECR_REGION to match the live cluster's region, e.g. Ohio:
#   CLUSTER=cairn-dsv4 ECR_REGION=us-east-2 bash infra/skypilot/cache-image.sh
# Creating a NEW region's repo needs ecr:CreateRepository (the cairn-s3-populate key has R/W on the existing
# cairn-sglang repo only) — this script ensure-creates it via the admin-cli profile, falling back to the
# write key; if both lack the perm it prints a clear message and you create the repo once from the console.
set -euo pipefail
CLUSTER="${CLUSTER:-cairn-dsv4}"
REGION="${ECR_REGION:-eu-south-2}"
ACCT="${ECR_ACCT:-<ACCOUNT_ID>}"   # set ECR_ACCT to your AWS account id
REPO="${ECR_REPO_NAME:-cairn-sglang}"
IMG="${SRC_IMAGE:-lmsysorg/sglang:deepseek-v4-blackwell}"
TAG="${ECR_TAG:-deepseek-v4-blackwell}"
ECR_URI="${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
: "${CAIRN_S3_WRITE_ACCESS_KEY_ID:?source infra/secrets.env first}"
echo "[cache-image] $CLUSTER: mirror $IMG -> $ECR_URI:$TAG (region $REGION)"

# Ensure the per-region ECR repo exists (idempotent) — a `docker push` to a missing repo fails. Prefer the
# admin-cli profile (can create); fall back to the populate key (can describe an existing repo, may not create).
echo "[cache-image] ensuring ECR repo $REPO exists in $REGION ..."
if aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" --profile admin-cli >/dev/null 2>&1 \
   || AWS_ACCESS_KEY_ID="$CAIRN_S3_WRITE_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$CAIRN_S3_WRITE_SECRET_ACCESS_KEY" \
        aws ecr describe-repositories --repository-names "$REPO" --region "$REGION" >/dev/null 2>&1; then
  echo "[cache-image] repo already present in $REGION"
elif aws ecr create-repository --repository-name "$REPO" --region "$REGION" --profile admin-cli >/dev/null 2>&1; then
  echo "[cache-image] created repo $REPO in $REGION (admin-cli)"
else
  echo "[cache-image] WARNING: repo $REPO missing in $REGION and could not auto-create (needs ecr:CreateRepository)."
  echo "[cache-image]          Create it once, then re-run:  aws ecr create-repository --repository-name $REPO --region $REGION"
  exit 1
fi
ssh -o StrictHostKeyChecking=no -o ConnectTimeout=25 "$CLUSTER" \
  "AWS_ACCESS_KEY_ID='$CAIRN_S3_WRITE_ACCESS_KEY_ID' AWS_SECRET_ACCESS_KEY='$CAIRN_S3_WRITE_SECRET_ACCESS_KEY' bash -s" <<REMOTE
set -euo pipefail
echo '[box] docker login to ECR ...'
aws ecr get-login-password --region $REGION | sudo docker login --username AWS --password-stdin $ECR_URI
echo '[box] tag + push (compressed registry content; in-region) ...'
sudo docker tag $IMG $ECR_URI:$TAG
sudo docker push $ECR_URI:$TAG
echo '[box] push complete'
REMOTE
echo "[cache-image] verifying ECR image ..."
aws ecr describe-images --repository-name "$REPO" --region "$REGION" --profile admin-cli \
  --query 'imageDetails[0].[imageTags[0],imageSizeInBytes]' --output text 2>/dev/null || \
  AWS_ACCESS_KEY_ID="$CAIRN_S3_WRITE_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$CAIRN_S3_WRITE_SECRET_ACCESS_KEY" \
  aws ecr describe-images --repository-name "$REPO" --region "$REGION" \
  --query 'imageDetails[0].[imageTags[0],imageSizeInBytes]' --output text
echo "[cache-image] done."
