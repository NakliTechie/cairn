#!/usr/bin/env bash
# Ensure the SHARED Cairn fleet security group exists (idempotent) — the fix for the self-replenish
# SG-isolation bug found live 2026-06-28.
#
# THE BUG: a replenished warm spare is launched as a SEPARATE SkyPilot cluster (cairn-dsv4-spare.sky.yaml
# via replenish-watcher.sh), so SkyPilot gives it its OWN per-cluster security group, isolated from the
# main fleet's SG. The recovery wire (ports 7777-7780: stage-in :7777, tail-sink :7779, spare-sink/announce
# :7780) is BIDIRECTIONAL between the fleet and the spare, so SG isolation broke BOTH halves:
#   • the spare's announce dial to the driver's :7780 hung in SYN-SENT (driver SG had no inbound from the
#     spare's SG), and
#   • after a manual ingress workaround, the actual recovery still failed at recovery.py:309
#     `sstate["read"].recv()` with TimeoutError — the predecessor stage's NEW data connection to the
#     spare's :7777 was blocked by the SPARE's own (separate-cluster) SG.
#
# THE FIX: pin BOTH cairn-dsv4.sky.yaml AND cairn-dsv4-spare.sky.yaml to ONE shared, named SG (`cairn-fleet`)
# via SkyPilot's `config.aws.security_group_name` (set in each yaml's `config:` block). When every box —
# the live fleet AND every replenished spare — shares one SG that permits the recovery ports across the VPC,
# all of 7777-7780 are mutually reachable with NO per-launch SG surgery. This script creates that SG with
# the right rules. Run it ONCE per region you intend to launch in, BEFORE the first launch.
#
# Security groups are per-(region, VPC). The SG NAME is the handle SkyPilot resolves at launch, so the same
# name (`cairn-fleet`) in each region's default VPC is what both yamls reference. Re-running is safe:
# find-or-create the SG, then authorize each rule ignoring "already exists" (InvalidPermission.Duplicate).
#
# Usage (with the cairn-skypilot key, which has CreateSecurityGroup + AuthorizeSecurityGroupIngress):
#   aws sso login --profile cairn-skypilot          # or `set -a; source infra/secrets.env; set +a`
#   bash infra/aws/ensure-fleet-sg.sh                # all four fleet regions
#   bash infra/aws/ensure-fleet-sg.sh us-east-2      # just one region
#
# Env overrides:
#   CAIRN_FLEET_SG_NAME   (default: cairn-fleet)     — must match `security_group_name` in BOTH yamls
#   CAIRN_SSH_CIDR        (default: 0.0.0.0/0)       — who may SSH in (SkyPilot provisioning needs :22)
set -uo pipefail

SG_NAME=${CAIRN_FLEET_SG_NAME:-cairn-fleet}
SSH_CIDR=${CAIRN_SSH_CIDR:-0.0.0.0/0}
RECOVERY_PORTS="7777-7780"        # stage-in :7777 · (:7778) · tail-sink :7779 · spare-sink/announce :7780
REGIONS=("$@")
[ ${#REGIONS[@]} -eq 0 ] && REGIONS=(eu-south-2 us-east-2 us-west-2 ap-northeast-2)

# Authorize one ingress rule, treating "already there" as success (idempotent re-runs).
authorize() {
  local region=$1 sg=$2; shift 2
  local out
  out=$(aws ec2 authorize-security-group-ingress --region "$region" --group-id "$sg" "$@" 2>&1) && return 0
  echo "$out" | grep -q "InvalidPermission.Duplicate" && return 0
  echo "[ensure-fleet-sg]   ! authorize failed: $out"; return 1
}

for REGION in "${REGIONS[@]}"; do
  echo "[ensure-fleet-sg] === $REGION ==="
  # Default VPC of the region (where SkyPilot launches by default). Override the SG's VPC by editing here
  # if you run the fleet in a non-default VPC — the SG must live in the SAME VPC the instances launch into.
  VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters Name=isDefault,Values=true \
             --query 'Vpcs[0].VpcId' --output text 2>/dev/null)
  if [ -z "$VPC_ID" ] || [ "$VPC_ID" = "None" ]; then
    echo "[ensure-fleet-sg]   ! no default VPC in $REGION — skipping (set one up or edit this script)"; continue
  fi
  VPC_CIDR=$(aws ec2 describe-vpcs --region "$REGION" --vpc-ids "$VPC_ID" \
               --query 'Vpcs[0].CidrBlock' --output text 2>/dev/null)
  echo "[ensure-fleet-sg]   vpc=$VPC_ID cidr=$VPC_CIDR"

  # Find-or-create the SG by name within this VPC.
  SG_ID=$(aws ec2 describe-security-groups --region "$REGION" \
            --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" \
            --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null)
  if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    SG_ID=$(aws ec2 create-security-group --region "$REGION" --vpc-id "$VPC_ID" \
              --group-name "$SG_NAME" \
              --description "Cairn shared fleet SG — recovery wire 7777-7780 reachable across the VPC (self-replenish spares join this same SG)" \
              --tag-specifications "ResourceType=security-group,Tags=[{Key=cairn,Value=true}]" \
              --query 'GroupId' --output text 2>/dev/null)
    [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ] && { echo "[ensure-fleet-sg]   ! create failed in $REGION"; continue; }
    echo "[ensure-fleet-sg]   created $SG_NAME = $SG_ID"
  else
    echo "[ensure-fleet-sg]   found  $SG_NAME = $SG_ID"
  fi

  # Rules: SSH for SkyPilot provisioning; the recovery wire from the VPC CIDR AND self-referencing (so the
  # range is reachable however the box resolves a peer — belt-and-suspenders for multi-subnet/peered VPCs).
  authorize "$REGION" "$SG_ID" --protocol tcp --port 22 --cidr "$SSH_CIDR"
  PROTO_PORT=(--protocol tcp --port "$RECOVERY_PORTS")
  authorize "$REGION" "$SG_ID" "${PROTO_PORT[@]}" --cidr "$VPC_CIDR"
  # Self-referencing rule via the IpPermissions form (cidr/source-group can't combine in the shorthand).
  out=$(aws ec2 authorize-security-group-ingress --region "$REGION" --group-id "$SG_ID" \
          --ip-permissions "IpProtocol=tcp,FromPort=7777,ToPort=7780,UserIdGroupPairs=[{GroupId=$SG_ID}]" 2>&1) \
    || echo "$out" | grep -q "InvalidPermission.Duplicate" \
    || echo "[ensure-fleet-sg]   ! self-ref authorize failed: $out"
  echo "[ensure-fleet-sg]   ok: :22 from $SSH_CIDR · :$RECOVERY_PORTS from $VPC_CIDR + self ($SG_NAME)"
done

echo "[ensure-fleet-sg] done. Both cairn-dsv4.sky.yaml and cairn-dsv4-spare.sky.yaml pin security_group_name=$SG_NAME."
