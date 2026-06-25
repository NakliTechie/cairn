#!/usr/bin/env bash
# Guarded + detached SkyPilot launch for Cairn.
#
# WHY THIS EXISTS (~/Code/infra/aws/README.md gotcha #9): SkyPilot runs ONE shared LOCAL api server per
# machine, and it provisions with its OWN ambient/cached AWS identity — NOT your client's AWS_PROFILE.
# On a box shared with mela/pitch/quorum, a naive `AWS_PROFILE=cairn-skypilot sky launch` can silently
# provision the box with ANOTHER project's key (whoever's profile started the server). That breaks
# scoped-key isolation and the tag-gated kill switch (the wrong key can't terminate a cairn=true box).
# This wrapper makes it deterministic:
#   0. assert the cairn profile's key really is cairn's (catch a mis-pasted sibling key),
#   1. restart the shared api server UNDER cairn's identity (the fresh server inherits AWS_PROFILE),
#   2. launch DETACHED (gotcha #10) so killing this client can't orphan/abort the run.
#
# NOTE: step 1 `sky api stop`s the shared server (mela/pitch/quorum use it too); it restarts on the next
# sky command under cairn. Run ONE project's fleet at a time on this machine.
#
# Usage:  bash infra/skypilot/launch.sh                                   # cairn-fleet, detached
#         CLUSTER=cairn-serve YAML=infra/skypilot/cairn-serve.sky.yaml bash infra/skypilot/launch.sh
#         bash infra/skypilot/launch.sh --env HF_TOKEN                    # extra args pass through to sky
# Monitor: AWS_PROFILE=cairn-skypilot <sky> logs <cluster>   (sky = your sky binary)
set -euo pipefail

PROFILE="${CAIRN_AWS_PROFILE:-cairn-skypilot}"
EXPECT_ARN="${CAIRN_EXPECT_ARN:-arn:aws:iam::615809814090:user/cairn-skypilot}"
CLUSTER="${CLUSTER:-${CAIRN_CLUSTER:-cairn-fleet}}"
YAML="${YAML:-${CAIRN_SKY_YAML:-infra/skypilot/cairn-fleet.sky.yaml}}"
IDLE="${CAIRN_IDLE_MIN:-60}"
SKY="${SKY:-$(command -v sky || echo "$HOME/Code/mela/.venv/bin/sky")}"

echo "[launch] sky=$SKY  profile=$PROFILE  cluster=$CLUSTER  yaml=$YAML"

# 0. the profile's access key must belong to cairn-skypilot (not a sibling key pasted by mistake)
got="$(aws sts get-caller-identity --profile "$PROFILE" --query Arn --output text 2>/dev/null || true)"
if [ "$got" != "$EXPECT_ARN" ]; then
  echo "[launch] ABORT: profile '$PROFILE' resolves to '${got:-<none>}', expected '$EXPECT_ARN'." >&2
  echo "[launch] Fix the profile (aws configure --profile $PROFILE) before launching." >&2
  exit 1
fi
echo "[launch] identity OK: $got"

# (~/Code/infra/aws/gpu-pricing-page.py) throw up the live GPU spot-pricing page to eyeball before launch
# — best-effort + opt-out (SKIP_PRICING_PAGE=1); never blocks the launch on failure.
PRICING_PAGE="${PRICING_PAGE:-$HOME/Code/infra/aws/gpu-pricing-page.py}"
if [ -z "${SKIP_PRICING_PAGE:-}" ] && [ -f "$PRICING_PAGE" ]; then
  echo "[launch] GPU spot-pricing page (eyeball before launch; SKIP_PRICING_PAGE=1 to skip) ..."
  AWS_PROFILE="$PROFILE" python3 "$PRICING_PAGE" --open >/dev/null 2>&1 || true
fi

# 1. force the shared local api server to (re)start under THIS identity (gotcha #9).
#    NOTE: this briefly stops the shared server used by mela/pitch/quorum — it restarts on the next
#    sky command. Run one project at a time on this machine.
echo "[launch] restarting SkyPilot api server under $PROFILE ..."
"$SKY" api stop >/dev/null 2>&1 || true
export AWS_PROFILE="$PROFILE"          # the fresh server inherits this — the actual fix
"$SKY" check aws >/dev/null 2>&1 || { echo "[launch] ABORT: 'sky check aws' failed under $PROFILE." >&2; exit 1; }

# 2. detached launch (gotcha #10) — the run lives on the cluster, independent of this client.
echo "[launch] launching DETACHED (-d, idle-autostop ${IDLE}m). The job survives this client exiting."
exec "$SKY" launch -c "$CLUSTER" "$YAML" -y -d -i "$IDLE" "$@"
