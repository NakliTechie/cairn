#!/usr/bin/env bash
# Host-side SELF-REPLENISH watcher — runs on rank0 of a live cairn-dsv4 fleet (OUTSIDE the container).
#
# serve_http (in the container) appends a line to $CAIRN_REPLENISH_REQ when a spare is consumed by a
# recovery (its --replenish-cmd = `echo provision >> /shared/cairn-replenish-req`, with /shared mounted from
# the host). This watcher launches a replacement 1-box spare cluster (cairn-dsv4-spare.sky.yaml) that dials
# back to this driver's spare-sink + --announces, refilling the warm pool toward warm_target.
#
# Requires on THIS box: the `sky` CLI + a SCOPED launch key (see infra/aws/cairn-spare-launch-policy.json).
# Why host-side (not in the container): keeps AWS creds OUT of the serving container; the box owns provisioning.
#
#   DRIVER_PRIVATE_IP=<rank0 ip> CAIRN_SPARE_RANK=<k> SKY_BIN=~/.cairn-sky-venv/bin/sky \
#     bash infra/skypilot/replenish-watcher.sh &
set -uo pipefail

REQ_FILE=${CAIRN_REPLENISH_REQ:-/tmp/cairn-shared/cairn-replenish-req}
DRIVER_HOST=${DRIVER_PRIVATE_IP:?set DRIVER_PRIVATE_IP to the rank0 private IP where spares dial back}
SPARE_SINK=${DRIVER_SPARE_SINK:-7780}
SPARE_RANK=${CAIRN_SPARE_RANK:-}
REGION=${CAIRN_REGION:-}
IMAGE=${CAIRN_IMAGE_ID:-}
SKY=${SKY_BIN:-sky}
SPARE_YAML=${CAIRN_SPARE_YAML:-infra/skypilot/cairn-dsv4-spare.sky.yaml}
MAX=${CAIRN_REPLENISH_MAX:-10}                 # safety cap on total replacements this watcher will launch
# FALLBACK for the SG-isolation bug (2026-06-28): the PRIMARY fix is to pin both cairn-dsv4.sky.yaml and
# cairn-dsv4-spare.sky.yaml to one shared named SG (security_group_name: cairn-fleet — see
# infra/aws/ensure-fleet-sg.sh). If you CAN'T pre-create a shared SG, set CAIRN_AUTHORIZE_RECOVERY_PORTS=1
# and this watcher will, after each launch, open the recovery ports (7777-7780) from the VPC CIDR on BOTH
# the driver's SG and the freshly-launched spare's SG. This is fragile (timing/per-launch) and needs
# ec2:AuthorizeSecurityGroupIngress on the launch key — prefer the shared SG.
AUTHORIZE_PORTS=${CAIRN_AUTHORIZE_RECOVERY_PORTS:-}
RECOVERY_PORTS="7777-7780"

# `sky launch` resolves the spare yaml's relative file_mounts (./fork, ./cairn_node, …) + the SPARE_YAML
# path from the CWD — so run from the synced project dir (the box's ~/.sky/file_mounts/cairn).
cd "${CAIRN_PROJECT_DIR:-.}" || { echo "[replenish-watcher] cannot cd to ${CAIRN_PROJECT_DIR:-.}"; exit 1; }

mkdir -p "$(dirname "$REQ_FILE")"; : > "$REQ_FILE"
echo "[replenish-watcher] watching $REQ_FILE -> launches $SPARE_YAML; driver=$DRIVER_HOST:$SPARE_SINK rank=${SPARE_RANK:-tail}"

region_args=""; [ -n "$REGION" ] && region_args="--region $REGION"; [ -n "$IMAGE" ] && region_args="$region_args --image-id $IMAGE"

# FALLBACK only (CAIRN_AUTHORIZE_RECOVERY_PORTS=1): open 7777-7780 from the VPC CIDR on both the driver's SG
# and the newly-launched spare's SG, so the bidirectional recovery wire works WITHOUT a shared SG. Best-effort
# + idempotent; the shared-SG fix (security_group_name) is strictly preferred. Needs AuthorizeSecurityGroupIngress.
authorize_recovery_ports() {
  command -v aws >/dev/null 2>&1 || { echo "[replenish-watcher] no aws CLI — can't authorize ports"; return 1; }
  local _T MAC R VPC_CIDR DRIVER_SG SPARE_SG i
  _T=$(curl -sX PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 120" 2>/dev/null)
  MAC=$(curl -s -H "X-aws-ec2-metadata-token: $_T" http://169.254.169.254/latest/meta-data/network/interfaces/macs/ 2>/dev/null | head -1 | tr -d /)
  R=$(curl -s -H "X-aws-ec2-metadata-token: $_T" http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null)
  VPC_CIDR=$(curl -s -H "X-aws-ec2-metadata-token: $_T" "http://169.254.169.254/latest/meta-data/network/interfaces/macs/$MAC/vpc-ipv4-cidr-block" 2>/dev/null)
  DRIVER_SG=$(curl -s -H "X-aws-ec2-metadata-token: $_T" "http://169.254.169.254/latest/meta-data/network/interfaces/macs/$MAC/security-group-ids" 2>/dev/null | head -1)
  [ -z "$VPC_CIDR" ] || [ -z "$R" ] && { echo "[replenish-watcher] IMDS lookup failed — skip authorize"; return 1; }
  _auth() { aws ec2 authorize-security-group-ingress --region "$R" --group-id "$1" \
              --protocol tcp --port "$RECOVERY_PORTS" --cidr "$VPC_CIDR" 2>&1 \
            | grep -vq "InvalidPermission.Duplicate" 2>/dev/null; return 0; }
  [ -n "$DRIVER_SG" ] && { _auth "$DRIVER_SG"; echo "[replenish-watcher] authorized :$RECOVERY_PORTS from $VPC_CIDR on driver SG $DRIVER_SG"; }
  # Poll for the spare instance (by the cairn-purpose label set in the spare yaml) to learn its SG.
  for i in $(seq 1 30); do
    SPARE_SG=$(aws ec2 describe-instances --region "$R" \
      --filters "Name=tag:cairn-purpose,Values=dsv4-flash-replacement-spare" "Name=instance-state-name,Values=pending,running" \
      --query 'Reservations[].Instances[].SecurityGroups[].GroupId' --output text 2>/dev/null | tr '\t' '\n' | sort -u | head -1)
    [ -n "$SPARE_SG" ] && [ "$SPARE_SG" != "None" ] && break
    sleep 10
  done
  if [ -n "$SPARE_SG" ] && [ "$SPARE_SG" != "None" ] && [ "$SPARE_SG" != "$DRIVER_SG" ]; then
    _auth "$SPARE_SG"; echo "[replenish-watcher] authorized :$RECOVERY_PORTS from $VPC_CIDR on spare SG $SPARE_SG"
  fi
}

launched=0
# -F: keep following across truncation/rotation; -n0: only NEW lines (ignore backlog at start).
tail -n0 -F "$REQ_FILE" 2>/dev/null | while read -r _line; do
  if [ "$launched" -ge "$MAX" ]; then echo "[replenish-watcher] hit CAIRN_REPLENISH_MAX=$MAX — ignoring"; continue; fi
  launched=$((launched + 1))
  TS=$(date +%s); CL="cairn-dsv4-spare-$TS"
  echo "[replenish-watcher] [$launched/$MAX] provisioning $CL ..."
  # shellcheck disable=SC2086
  $SKY launch -c "$CL" "$SPARE_YAML" -y -d --down $region_args \
    --env DRIVER_HOST="$DRIVER_HOST" --env DRIVER_SPARE_SINK="$SPARE_SINK" \
    --env CAIRN_SPARE_RANK="$SPARE_RANK" --env HF_TOKEN --env SHARD_PSK \
    --env CAIRN_S3_CACHE_ACCESS_KEY_ID --env CAIRN_S3_CACHE_SECRET_ACCESS_KEY \
    --env CAIRN_S3_WRITE_ACCESS_KEY_ID --env CAIRN_S3_WRITE_SECRET_ACCESS_KEY \
    > "/tmp/replenish-$TS.log" 2>&1 \
    && echo "[replenish-watcher] $CL launched (it will dial back + announce when warm)" \
    || echo "[replenish-watcher] $CL FAILED — see /tmp/replenish-$TS.log"
  # FALLBACK path only — no-op when both yamls share security_group_name (the spare is already reachable).
  [ -n "$AUTHORIZE_PORTS" ] && authorize_recovery_ports
done
