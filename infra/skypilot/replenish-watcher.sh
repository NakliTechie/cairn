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

# `sky launch` resolves the spare yaml's relative file_mounts (./fork, ./cairn_node, …) + the SPARE_YAML
# path from the CWD — so run from the synced project dir (the box's ~/.sky/file_mounts/cairn).
cd "${CAIRN_PROJECT_DIR:-.}" || { echo "[replenish-watcher] cannot cd to ${CAIRN_PROJECT_DIR:-.}"; exit 1; }

mkdir -p "$(dirname "$REQ_FILE")"; : > "$REQ_FILE"
echo "[replenish-watcher] watching $REQ_FILE -> launches $SPARE_YAML; driver=$DRIVER_HOST:$SPARE_SINK rank=${SPARE_RANK:-tail}"

region_args=""; [ -n "$REGION" ] && region_args="--region $REGION"; [ -n "$IMAGE" ] && region_args="$region_args --image-id $IMAGE"
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
done
