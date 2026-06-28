#!/usr/bin/env bash
# Build a Cairn "warm AMI" — a per-region AMI that bakes the DeepSeek-V4-Flash FP8 checkpoint
# (~294 GB) AND the sglang Blackwell Docker image (~82 GB) onto an EBS DATA VOLUME, so a
# replacement warm spare launched from it skips the ~30-min S3->NVMe weight sync + image pull.
# Replenishment drops ~30-40 min -> ~5-8 min (with FSR). See infra/skypilot/WARM-AMI.md for the
# full design, the warm-up breakdown, and the cost/FSR tradeoffs.
#
# !!! THIS SCRIPT IS WRITTEN BUT NOT MEANT TO BE RUN BLINDLY !!!
# It provisions a real builder instance + a 400 GB EBS volume, stages ~376 GB onto it, snapshots
# it, and registers an AMI. It uses ADMIN-TIER EC2 perms (run-instances, create-image,
# copy-image, FSR) that the least-priv cairn-skypilot key does NOT have by design (see
# infra/aws/README.md credential posture) — hence AWS_PROFILE defaults to admin-cli. Read it,
# understand the cost, then run a phase at a time. Nothing here tears down on its own.
#
# WHY EBS not NVMe: the g7e NVMe at /opt/dlami/nvme is INSTANCE STORE (ephemeral) and is NOT
# captured in an AMI — AMIs only snapshot EBS. So the weights must be baked onto EBS to survive
# into the image. (At runtime the warm spare reads its slice from the FSR'd EBS volume; an
# optional background copy moves weights EBS->NVMe for fast reshape — off the announce path.)
#
# Usage (phase at a time):
#   set -a; source infra/secrets.env; set +a          # for HF_TOKEN + CAIRN_S3_*/ECR creds passthrough
#   REGION=us-east-2 AZ=us-east-2a bash infra/skypilot/build-warm-ami.sh provision   # launch builder + volume
#   REGION=us-east-2 bash infra/skypilot/build-warm-ami.sh stage    # ssh in, bake image+weights onto the volume
#   REGION=us-east-2 bash infra/skypilot/build-warm-ami.sh image    # stop builder, create-image, print AMI id
#   REGION=us-east-2 bash infra/skypilot/build-warm-ami.sh copy --to us-west-2,eu-south-2   # replicate per region
#   REGION=us-east-2 AZ=us-east-2a SNAP=snap-xxxx bash infra/skypilot/build-warm-ami.sh fsr-enable   # at RUN time
#   REGION=us-east-2 AZ=us-east-2a SNAP=snap-xxxx bash infra/skypilot/build-warm-ami.sh fsr-disable  # at teardown
#   REGION=us-east-2 bash infra/skypilot/build-warm-ami.sh cleanup  # terminate the builder when done
set -uo pipefail

# ---- config (override via env) ---------------------------------------------------------------
REGION="${REGION:?set REGION, e.g. us-east-2}"
AZ="${AZ:-${REGION}a}"                                   # FSR + EBS are AZ-local; pin the build AZ
PROFILE="${AWS_PROFILE:-admin-cli}"                      # admin-tier perms (cairn-skypilot can't do these)
AWS="aws --region $REGION --profile $PROFILE"

# The DLAMI base per region (same build as cairn-dsv4.sky.yaml's image_id). Keep these in sync.
declare -A DLAMI=(
  [eu-south-2]=ami-03cc3ae6c38cc7b29
  [us-east-2]=ami-02584dd48f685760d
  [ap-northeast-2]=ami-02f2234b443117395
)
BASE_AMI="${BASE_AMI:-${DLAMI[$REGION]:-}}"
INSTANCE_TYPE="${INSTANCE_TYPE:-g7e.2xlarge}"            # DLAMI base; a GPU isn't needed to STAGE, but
                                                        # keeping the prod instance family keeps root identical
DATA_GB="${DATA_GB:-400}"                                # holds 294 GB weights + 82 GB image, with headroom
DATA_VOL_TYPE="${DATA_VOL_TYPE:-gp3}"                    # gp3 (1 GB/s) ample for the one-time slice load;
                                                        # io2 (Block Express) if you want NVMe-class reshape reads
KEY_NAME="${KEY_NAME:-cairn-warm-ami-build}"
SG_ID="${SG_ID:-}"                                       # security group allowing your SSH; required for `provision`
SUBNET_ID="${SUBNET_ID:-}"                               # a subnet IN $AZ; required for `provision`
TAG="cairn=true,cairn-purpose=warm-ami-build"
MODEL_S3="${MODEL_S3:-s3://skypilot-cairn-weights-${REGION}/deepseek-v4-flash-fp8/main}"
IMG="${IMG:-lmsysorg/sglang:deepseek-v4-blackwell}"
ECR_ACCT="${ECR_ACCT:-<ACCOUNT_ID>}"   # set ECR_ACCT to your AWS account id
ECR_REPO="${ECR_REPO:-cairn-sglang}"
ECR_URI="${ECR_ACCT}.dkr.ecr.${REGION}.amazonaws.com/${ECR_REPO}"
STATE_DIR="${STATE_DIR:-/tmp/cairn-warm-ami-${REGION}}"  # stashes instance/volume/ami ids between phases
mkdir -p "$STATE_DIR"

say() { echo "[warm-ami][$REGION] $*"; }
die() { echo "[warm-ami][$REGION] FATAL: $*" >&2; exit 1; }
load() { [ -f "$STATE_DIR/$1" ] && cat "$STATE_DIR/$1"; }
save() { printf '%s' "$2" > "$STATE_DIR/$1"; }

phase="${1:-help}"; shift || true

case "$phase" in

# --- 1. provision: launch a builder from the DLAMI with a blank data volume in $AZ -----------
provision)
  [ -n "$BASE_AMI" ] || die "no DLAMI for $REGION — set BASE_AMI"
  [ -n "$SG_ID" ] && [ -n "$SUBNET_ID" ] || die "set SG_ID + SUBNET_ID (a subnet in $AZ that allows your SSH)"
  say "launching builder: $INSTANCE_TYPE from $BASE_AMI in $AZ, + ${DATA_GB}GB $DATA_VOL_TYPE data volume"
  # Block device mapping: keep the DLAMI root, add /dev/sdf as the warm-data volume (blank for now).
  bdm="[{\"DeviceName\":\"/dev/sdf\",\"Ebs\":{\"VolumeSize\":${DATA_GB},\"VolumeType\":\"${DATA_VOL_TYPE}\",\"DeleteOnTermination\":true}}]"
  iid=$($AWS ec2 run-instances \
    --image-id "$BASE_AMI" --instance-type "$INSTANCE_TYPE" \
    --key-name "$KEY_NAME" --security-group-ids "$SG_ID" --subnet-id "$SUBNET_ID" \
    --block-device-mappings "$bdm" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=cairn,Value=true},{Key=cairn-purpose,Value=warm-ami-build}]" \
    --query 'Instances[0].InstanceId' --output text) || die "run-instances failed"
  save instance "$iid"; say "builder instance: $iid (waiting for running + status ok ...)"
  $AWS ec2 wait instance-status-ok --instance-ids "$iid" || die "instance never became healthy"
  ip=$($AWS ec2 describe-instances --instance-ids "$iid" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  save ip "$ip"; say "builder ready at $ip — next: 'stage' (ssh in + bake)"
  ;;

# --- 2. stage: bake the docker image + weights onto the data volume --------------------------
stage)
  ip=$(load ip) || die "no builder ip — run 'provision' first"
  : "${HF_TOKEN:?source infra/secrets.env (HF_TOKEN + CAIRN_* creds needed for the passthrough)}"
  say "staging onto the data volume via ssh ($ip) — this moves ~376 GB, expect ~25-40 min"
  # The remote heredoc: format + mount the data volume, point docker at it, pull the image from the
  # in-region ECR mirror, sync weights S3->volume, restore the kernel, and persist the mount in fstab.
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=30 ubuntu@"$ip" \
    "REGION='$REGION' ECR_URI='$ECR_URI' IMG='$IMG' MODEL_S3='$MODEL_S3' \
     CAIRN_S3_CACHE_ACCESS_KEY_ID='${CAIRN_S3_CACHE_ACCESS_KEY_ID:-}' CAIRN_S3_CACHE_SECRET_ACCESS_KEY='${CAIRN_S3_CACHE_SECRET_ACCESS_KEY:-}' \
     CAIRN_S3_WRITE_ACCESS_KEY_ID='${CAIRN_S3_WRITE_ACCESS_KEY_ID:-}' CAIRN_S3_WRITE_SECRET_ACCESS_KEY='${CAIRN_S3_WRITE_SECRET_ACCESS_KEY:-}' \
     HF_TOKEN='${HF_TOKEN:-}' bash -s" <<'REMOTE'
set -euo pipefail
WARM=/opt/cairn-warm
# The blank data volume is the non-root NVMe-backed EBS at /dev/sdf -> shows as /dev/nvme1n1 on Nitro.
DEV=$(lsblk -dpno NAME,SIZE | awk '$2 ~ /G$/ && $1 !~ /nvme0/ {print $1; exit}')
DEV=${DEV:-/dev/nvme1n1}
echo "[box] data volume = $DEV"
if ! blkid "$DEV" >/dev/null 2>&1; then echo "[box] mkfs.ext4 $DEV"; sudo mkfs.ext4 -F "$DEV"; fi
sudo mkdir -p "$WARM"; sudo mount "$DEV" "$WARM"; sudo chown "$USER":"$USER" "$WARM"
sudo mkdir -p "$WARM/model" "$WARM/docker"

# docker data-root -> the data volume (merge, don't clobber the DLAMI's nvidia runtime block)
command -v docker >/dev/null || { curl -fsSL https://get.docker.com | sudo sh; sudo systemctl enable --now docker; }
sudo python3 -c "import json,os;p='/etc/docker/daemon.json';d=(json.load(open(p)) if os.path.exists(p) and os.path.getsize(p)>0 else {});d['data-root']='$WARM/docker';json.dump(d,open(p,'w'),indent=2)"
sudo systemctl restart docker || true

# image: pull from the in-region ECR mirror (compressed ~25 GB, AWS backbone), retag to canonical name
if ! sudo docker image inspect "$IMG" >/dev/null 2>&1; then
  if [ -n "${CAIRN_S3_WRITE_ACCESS_KEY_ID:-}" ] \
     && AWS_ACCESS_KEY_ID="$CAIRN_S3_WRITE_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$CAIRN_S3_WRITE_SECRET_ACCESS_KEY" \
          aws ecr get-login-password --region "$REGION" 2>/dev/null \
          | sudo docker login --username AWS --password-stdin "$ECR_URI" >/dev/null 2>&1 \
     && sudo docker pull "$ECR_URI:deepseek-v4-blackwell"; then
    sudo docker tag "$ECR_URI:deepseek-v4-blackwell" "$IMG"; echo "[box] image baked from ECR mirror"
  else
    echo "[box] ECR miss -> Docker Hub pull (~82 GB)"; sudo docker pull "$IMG"
  fi
fi

# weights: in-region S3 sync onto the data volume
if [ -n "${CAIRN_S3_CACHE_ACCESS_KEY_ID:-}" ] \
   && AWS_ACCESS_KEY_ID="$CAIRN_S3_CACHE_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$CAIRN_S3_CACHE_SECRET_ACCESS_KEY" \
        aws s3 ls "$MODEL_S3/config.json" --region "$REGION" >/dev/null 2>&1; then
  echo "[box] S3 HIT -> baking weights onto $WARM/model"
  AWS_ACCESS_KEY_ID="$CAIRN_S3_CACHE_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$CAIRN_S3_CACHE_SECRET_ACCESS_KEY" \
    aws s3 sync "$MODEL_S3" "$WARM/model" --region "$REGION" --only-show-errors
else
  echo "[box] S3 MISS -> HF (populate the in-region bucket first; see CACHING.md)"
  export HF_HUB_ENABLE_HF_TRANSFER=1; pip3 install -q --user "huggingface_hub[cli]" hf_transfer
  python3 -c "from huggingface_hub import snapshot_download; snapshot_download('sgl-project/DeepSeek-V4-Flash-FP8', local_dir='$WARM/model', max_workers=8)"
fi

# 0xSero kernel (tiny; bake it too so the warm box skips even the cache restore)
if [ ! -d ~/dsv4 ]; then git clone https://github.com/0xSero/deepseek-v4-flash-sm120 ~/dsv4; fi
# persist the mount so a box launched from this AMI auto-mounts the volume on boot
UUID=$(sudo blkid -s UUID -o value "$DEV")
echo "UUID=$UUID $WARM ext4 defaults,nofail 0 2" | sudo tee -a /etc/fstab
sync
echo "[box] staged: $(du -sh "$WARM/model" | cut -f1) weights + image on $WARM (fstab persisted)"
REMOTE
  say "stage complete — next: 'image' (stop + create-image)"
  ;;

# --- 3. image: stop the builder, create an AMI capturing root + the warm-data volume ---------
image)
  iid=$(load instance) || die "no builder instance — run 'provision' first"
  say "stopping builder $iid for a consistent snapshot ..."
  $AWS ec2 stop-instances --instance-ids "$iid" >/dev/null
  $AWS ec2 wait instance-stopped --instance-ids "$iid" || die "instance never stopped"
  name="cairn-warm-dsv4-${REGION}-$(load ts 2>/dev/null || echo build)"
  say "create-image $name (snapshots DLAMI-root + warm-data volume; registers AMI w/ BDM) ..."
  ami=$($AWS ec2 create-image --instance-id "$iid" --name "$name" \
    --description "Cairn warm AMI: DSV4-Flash FP8 weights + sglang image baked on EBS data volume" \
    --tag-specifications "ResourceType=image,Tags=[{Key=cairn,Value=true},{Key=cairn-purpose,Value=warm-ami}]" \
    --query 'ImageId' --output text) || die "create-image failed"
  save ami "$ami"; say "AMI registering: $ami (waiting for available ...)"
  $AWS ec2 wait image-available --image-ids "$ami" || die "AMI never became available"
  # surface the data-volume snapshot id (you enable FSR on THIS, per AZ, at run time)
  snap=$($AWS ec2 describe-images --image-ids "$ami" \
    --query 'Images[0].BlockDeviceMappings[?DeviceName==`/dev/sdf`].Ebs.SnapshotId | [0]' --output text)
  save snapshot "$snap"
  say "DONE. Warm AMI for $REGION: $ami   (data-volume snapshot: $snap)"
  say "  -> put '$ami' in cairn-dsv4-spare.sky.yaml image_id for $REGION (and the main yaml)"
  say "  -> at RUN time, enable FSR on $snap in your fleet AZ: bash $0 fsr-enable (SNAP=$snap AZ=...)"
  ;;

# --- 4. copy: replicate the AMI into other regions -------------------------------------------
copy)
  ami=$(load ami) || die "no AMI — run 'image' first"
  [ "${1:-}" = "--to" ] || die "usage: ... copy --to us-west-2,eu-south-2"
  IFS=',' read -ra DESTS <<< "$2"
  for d in "${DESTS[@]}"; do
    say "copy-image $ami ($REGION) -> $d ..."
    nid=$(aws --region "$d" --profile "$PROFILE" ec2 copy-image \
      --source-region "$REGION" --source-image-id "$ami" \
      --name "cairn-warm-dsv4-${d}" \
      --description "Cairn warm AMI (copied from $REGION)" \
      --query 'ImageId' --output text) || { say "copy to $d FAILED"; continue; }
    say "  $d AMI: $nid (becomes available async; check: aws --region $d ec2 wait image-available --image-ids $nid)"
  done
  ;;

# --- 5. fsr-enable / fsr-disable: the per-AZ, hourly-billed fast-restore toggle ---------------
# Enable at fleet bring-up (reaches 'enabled' within ~60 min), DISABLE at teardown. $0.75/hr/AZ.
fsr-enable)
  snap="${SNAP:-$(load snapshot)}"; [ -n "$snap" ] || die "set SNAP=snap-xxxx (the data-volume snapshot)"
  say "enabling FSR on $snap in $AZ (\$0.75/hr while enabled; ~60 min to reach 'enabled') ..."
  $AWS ec2 enable-fast-snapshot-restores --availability-zones "$AZ" --source-snapshot-ids "$snap" \
    --query 'Successful' --output json || die "enable-fast-snapshot-restores failed"
  say "FSR enabling. Poll: $AWS ec2 describe-fast-snapshot-restores --filters Name=snapshot-id,Values=$snap"
  ;;

fsr-disable)
  snap="${SNAP:-$(load snapshot)}"; [ -n "$snap" ] || die "set SNAP=snap-xxxx"
  say "disabling FSR on $snap in $AZ (stops the hourly charge) ..."
  $AWS ec2 disable-fast-snapshot-restores --availability-zones "$AZ" --source-snapshot-ids "$snap" \
    --query 'Successful' --output json || die "disable failed"
  say "FSR disabled in $AZ."
  ;;

# --- 6. cleanup: terminate the builder (the AMI + its snapshot persist) -----------------------
cleanup)
  iid=$(load instance) || die "no builder instance recorded"
  say "terminating builder $iid (AMI + snapshot persist; data volume DeleteOnTermination=true) ..."
  $AWS ec2 terminate-instances --instance-ids "$iid" >/dev/null && say "terminated $iid"
  ;;

*)
  cat <<EOF
Cairn warm-AMI builder — phases (run in order; see top-of-file usage + WARM-AMI.md):
  provision    launch a DLAMI builder + blank ${DATA_GB}GB data volume in \$AZ   (set SG_ID + SUBNET_ID)
  stage        ssh in; bake sglang image + ~294GB FP8 weights onto the volume
  image        stop builder; create-image -> per-region warm AMI (+ data-volume snapshot)
  copy --to R1,R2   replicate the AMI into other regions
  fsr-enable   (RUN time) enable Fast Snapshot Restore on the snapshot in \$AZ   (\$0.75/hr/AZ)
  fsr-disable  (teardown) disable FSR to stop the hourly charge
  cleanup      terminate the builder instance (AMI + snapshot persist)
Env: REGION (req), AZ, AWS_PROFILE=admin-cli, SG_ID, SUBNET_ID, DATA_GB, DATA_VOL_TYPE, SNAP.
EOF
  ;;
esac
