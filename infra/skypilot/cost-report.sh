#!/usr/bin/env bash
# Cairn — pull ACTUAL GPU spend from AWS Cost Explorer, filtered by the `cairn` tag. This is the source
# of truth for the cost claim (docs/cost-methodology.md); the local launchlog ledger reconciles against it.
#
# PREREQUISITES (one-time):
#   1. Activate the cost-allocation tags: AWS Billing console → Cost allocation tags → activate `cairn`,
#      `cairn-purpose`, `cairn-model`. Takes ~24h to start populating after activation.
#   2. Billing / Cost-Explorer permission (ce:GetCostAndUsage) — usually the ADMIN identity, NOT the
#      scoped cairn-skypilot key. Run with admin creds (AWS_PROFILE=admin … or exported keys).
#
# Usage:
#   cost-report.sh                 # month-to-date, grouped by cairn-purpose (spot-proof vs ondemand-baseline)
#   cost-report.sh 2026-06-01 2026-06-30
#   cost-report.sh --by-model 2026-06-01 2026-06-30   # group by cairn-model instead
set -euo pipefail

GROUP_KEY="cairn-purpose"
if [ "${1:-}" = "--by-model" ]; then GROUP_KEY="cairn-model"; shift; fi

START="${1:-$(date -u +%Y-%m-01)}"
# Cost Explorer's End is EXCLUSIVE → default to tomorrow so today is included. (macOS date, then GNU date.)
END="${2:-$(date -u -v+1d +%Y-%m-%d 2>/dev/null || date -u -d tomorrow +%Y-%m-%d)}"

echo "Cairn actual spend  ${START} → ${END} (End exclusive)  ·  UnblendedCost, grouped by ${GROUP_KEY}"
echo "(needs the cairn cost-allocation tag activated + billing perms — see header)"
echo

aws ce get-cost-and-usage \
  --time-period "Start=${START},End=${END}" \
  --granularity MONTHLY \
  --metrics UnblendedCost \
  --filter '{"Tags":{"Key":"cairn","Values":["true"]}}' \
  --group-by "Type=TAG,Key=${GROUP_KEY}" \
  --output table

echo
echo "Reconcile against the local ledger:  python infra/skypilot/launchlog.py report"
echo "Cost-claim methodology (what counts, smell-test rules): docs/cost-methodology.md"
