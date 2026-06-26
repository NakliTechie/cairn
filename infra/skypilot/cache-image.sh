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
set -euo pipefail
CLUSTER="${CLUSTER:-cairn-dsv4}"
REGION="${ECR_REGION:-eu-south-2}"
ACCT="${ECR_ACCT:-615809814090}"
REPO="${ECR_REPO_NAME:-cairn-sglang}"
IMG="${SRC_IMAGE:-lmsysorg/sglang:deepseek-v4-blackwell}"
TAG="${ECR_TAG:-deepseek-v4-blackwell}"
ECR_URI="${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${REPO}"
: "${CAIRN_S3_WRITE_ACCESS_KEY_ID:?source infra/secrets.env first}"
echo "[cache-image] $CLUSTER: mirror $IMG -> $ECR_URI:$TAG (region $REGION)"
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
