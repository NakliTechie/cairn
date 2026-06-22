#!/usr/bin/env bash
# nuke.sh — the Cairn cost kill switch + EC2 ground-truth check.
#
# Lists (default) or TERMINATES (--force) every cairn=true EC2 instance across Cairn's regions, talking
# to the EC2 API DIRECTLY — the one teardown that works even when `sky down` wedges on an INIT cluster
# (the 2026-06-22 cost crisis: a launch wedged in INIT left 3 L4s billing and `sky down` couldn't kill
# them). EC2 is ground truth; `sky status` can lie.
#
#   infra/skypilot/nuke.sh            # ground-truth check: what cairn instances are running? (no changes)
#   infra/skypilot/nuke.sh --force    # TERMINATE them all (break glass)
#
# Safe: read-only by default; the scoped IAM key only permits terminating cairn=true-tagged instances.
set -uo pipefail

PROFILE=${AWS_PROFILE:-cairn-skypilot}
REGIONS=${CAIRN_REGIONS:-"eu-south-2 ap-northeast-2"}   # where Cairn launches (mutation-locked to these)
STATES="running,pending,stopping,stopped"              # everything that bills compute and/or EBS
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1

command -v aws >/dev/null 2>&1 || { echo "nuke: aws CLI not found"; exit 2; }

found=0
for r in $REGIONS; do
  ids=$(aws --profile "$PROFILE" ec2 describe-instances --region "$r" \
        --filters "Name=tag:cairn,Values=true" "Name=instance-state-name,Values=$STATES" \
        --query 'Reservations[].Instances[].InstanceId' --output text 2>/dev/null) || {
          echo "[$r] describe failed (profile '$PROFILE' / creds?)"; continue; }
  [ -z "$ids" ] && continue
  found=1
  echo "[$r] cairn instances (id / type / state / launched):"
  aws --profile "$PROFILE" ec2 describe-instances --region "$r" --instance-ids $ids \
      --query 'Reservations[].Instances[].[InstanceId,InstanceType,State.Name,LaunchTime]' --output text 2>/dev/null
  if [ "$FORCE" = "1" ]; then
    echo "[$r] *** TERMINATING ***"
    aws --profile "$PROFILE" ec2 terminate-instances --region "$r" --instance-ids $ids \
        --query 'TerminatingInstances[].[InstanceId,CurrentState.Name]' --output text 2>&1
  fi
done

if [ "$found" = "0" ]; then
  echo "No cairn instances in [$REGIONS] — nothing billing ✓"
elif [ "$FORCE" = "0" ]; then
  echo
  echo ">>> DRY RUN. Re-run with --force to TERMINATE the above:  infra/skypilot/nuke.sh --force"
fi
