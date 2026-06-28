#!/usr/bin/env bash
# Reliable Cairn teardown using cairn's OWN scoped key.
#
# WHY: if the shared api server is running under a DIFFERENT
# identity, `sky down` can't terminate a cairn-tagged box (the wrong key's tag-gate blocks it).
# Terminating directly with cairn's key works via the `cairn=true` tag gate regardless of the server's
# identity. Then we clear the local SkyPilot record with --purge.
#
# This is the cluster-aware teardown (terminate by tag across ALL cairn regions, then purge the sky
# record). `nuke.sh` remains the break-glass EC2 ground-truth check/kill (read-only by default).
#
# Usage:  bash infra/skypilot/teardown.sh                       # cairn-fleet
#         CLUSTER=cairn-serve bash infra/skypilot/teardown.sh
set -euo pipefail

PROFILE="${CAIRN_AWS_PROFILE:-cairn-skypilot}"
CLUSTER="${CLUSTER:-${CAIRN_CLUSTER:-cairn-fleet}}"
SKY="${SKY:-$(command -v sky)}"
REGIONS="${CAIRN_REGIONS:-eu-south-2 ap-northeast-2 us-east-2 us-west-2}"

for r in $REGIONS; do
  ids="$(aws ec2 describe-instances --profile "$PROFILE" --region "$r" \
    --filters "Name=tag:cairn,Values=true" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query "Reservations[].Instances[].InstanceId" --output text 2>/dev/null || true)"
  if [ -n "$ids" ]; then
    echo "[teardown] $r: terminating $ids"
    aws ec2 terminate-instances --profile "$PROFILE" --region "$r" --instance-ids $ids >/dev/null
  else
    echo "[teardown] $r: no cairn instances"
  fi
done

echo "[teardown] clearing local SkyPilot record for $CLUSTER"
AWS_PROFILE="$PROFILE" "$SKY" down "$CLUSTER" -y --purge >/dev/null 2>&1 || true
echo "[teardown] done — verify: aws ec2 describe-instances --profile $PROFILE --region <r> --filters Name=tag:cairn,Values=true"
