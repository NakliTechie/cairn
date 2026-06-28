#!/usr/bin/env bash
# bringup.sh — ONE-COMMAND Cairn fleet bring-up (DeepSeek-V4-Flash FP8, layer-split + warm-spare recovery).
#
# Consolidates the previously-manual dance documented in cairn-dsv4.sky.yaml's header into a single
# parameterized flow:
#   1. load secrets correctly (set -a; source infra/secrets.env; set +a — the documented gotcha, see below)
#   2. (optionally) sweep spot prices so you launch in the cheapest region
#   3. launch the fleet via the identity-safe launch.sh wrapper (CAIRN_NWAY active + warm spares)
#   4. wait for the rank0 serve_http endpoint to bind :8000
#   5. open the SSH tunnel to localhost:8000
#   6. print how to chat + how to tear down
#
# Everything is env-var parameterized with defaults matching the current LIVE config
# (DeepSeek-V4-Flash FP8, NWAY=4, 6 nodes = 4 active + 2 spares, us-east-2). Override any knob inline:
#   NWAY=3 NUM_NODES=5 REGION=eu-south-2 bash infra/skypilot/bringup.sh
#
# Quick start (from the repo root):
#   cp infra/secrets.env.example infra/secrets.env   # then fill HF_TOKEN, SHARD_PSK, CAIRN_S3_CACHE_*, ...
#   bash infra/skypilot/bringup.sh                    # launch + tunnel + print chat/teardown instructions
#
# THIS DOES NOT TOUCH A RUNNING FLEET. It launches a NEW cluster named $CLUSTER (default cairn-dsv4).
# Teardown is a one-liner (see the end of this script / docs/operator-README.md):
#   infra/skypilot/nuke.sh --force      # terminate every cairn=true box (EC2-API ground truth)
#
# Full operator guide: docs/operator-README.md
set -uo pipefail

# ---------------------------------------------------------------------------------------------------
# Resolve the repo root so this runs from anywhere, and `cd` there: sky resolves the yaml's relative
# file_mounts (./fork, ./cairn_node, …) + the yaml path from the CWD.
# ---------------------------------------------------------------------------------------------------
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT" || { echo "[bringup] cannot cd to repo root $ROOT"; exit 1; }

# ---------------------------------------------------------------------------------------------------
# Knobs (defaults = the current live config). Override any inline as VAR=value before the command.
# ---------------------------------------------------------------------------------------------------
MODEL="${MODEL:-sgl-project/DeepSeek-V4-Flash-FP8}"   # served name + the HF/S3 weight id (run: MODEL=...)
NWAY="${NWAY:-4}"                                      # ACTIVE layer-split stages (FP8 needs 4; 3 OOMs)
NUM_NODES="${NUM_NODES:-6}"                            # total boxes = NWAY active + (NUM_NODES-NWAY) spares
REGION="${REGION:-us-east-2}"                          # us-east-2 (Ohio) = cheapest g7e spot (2026-06-24)
CLUSTER="${CLUSTER:-cairn-dsv4}"                       # sky cluster name (also the `ssh <name>` host alias)
YAML="${YAML:-infra/skypilot/cairn-dsv4.sky.yaml}"
API_KEY="${CAIRN_API_KEY:-sk-cairn-demo}"             # BYOK Bearer key the endpoint enforces
LOCAL_PORT="${LOCAL_PORT:-8000}"                       # local port the tunnel binds (chat page + curl)
IDLE_MIN="${CAIRN_IDLE_MIN:-60}"                       # sky idle-autostop minutes (passed to launch.sh)
WARM_TARGET="${WARM_TARGET:-$((NUM_NODES - NWAY))}"    # warm spares to keep; default = the spare count.
                                                       #   serve_http derives --warm-target/--initial-spares
                                                       #   from (num_nodes - NWAY) inside the yaml, so this is
                                                       #   informational unless you also reshape the yaml.
SELF_REPLENISH="${CAIRN_SELF_REPLENISH:-}"            # non-empty → provision a replacement on consumption
DO_SWEEP="${DO_SWEEP:-1}"                              # 1 = run spot_sweep.py first (read-only price check)
AUTO_TUNNEL="${AUTO_TUNNEL:-1}"                        # 1 = open the ssh -L tunnel after the endpoint binds

# Per-region Blackwell DLAMI (R580 driver) — REQUIRED for sm_120. AMIs are per-region; the yaml's default
# matches eu-south-2, so when REGION differs we MUST override --image-id too. (Known 20260619-build ids,
# kept in sync with cairn-dsv4.sky.yaml's resources comment.)
declare -A DLAMI=(
  [eu-south-2]="ami-03cc3ae6c38cc7b29"      # Spain (matches the yaml default)
  [us-east-2]="ami-02584dd48f685760d"       # Ohio (cheapest g7e)
  [ap-northeast-2]="ami-02f2234b443117395"  # Seoul
)
IMAGE_ID="${IMAGE_ID:-${DLAMI[$REGION]:-}}"

PROFILE="${CAIRN_AWS_PROFILE:-cairn-skypilot}"
SKY="${SKY:-$(command -v sky)}"

echo "[bringup] model=$MODEL  nway=$NWAY  nodes=$NUM_NODES (=$NWAY active + $((NUM_NODES-NWAY)) spares)"
echo "[bringup] region=$REGION  image=$IMAGE_ID  cluster=$CLUSTER  idle-autostop=${IDLE_MIN}m"
echo "[bringup] endpoint key=$API_KEY  local tunnel port=$LOCAL_PORT  self-replenish=${SELF_REPLENISH:-off}"

# ---------------------------------------------------------------------------------------------------
# 1. Secrets — THE GOTCHA: `source infra/secrets.env` alone does NOT export the vars, so the `sky`
#    subprocess (and the --env passthroughs below) can't see them. `set -a` marks every assignment for
#    export for the duration; `set +a` restores. secrets.env is bare VAR=value (gitignored); the repo
#    ships only secrets.env.example. (launch.py parses the same file with k,v=line.split("=",1).)
# ---------------------------------------------------------------------------------------------------
if [ ! -f infra/secrets.env ]; then
  echo "[bringup] ABORT: infra/secrets.env not found. Run: cp infra/secrets.env.example infra/secrets.env" >&2
  echo "[bringup] then fill HF_TOKEN, SHARD_PSK, and (for the in-region cache) CAIRN_S3_CACHE_* / CAIRN_S3_WRITE_*." >&2
  exit 1
fi
set -a; source infra/secrets.env; set +a

if [ -z "${HF_TOKEN:-}" ]; then
  echo "[bringup] ABORT: HF_TOKEN is empty in infra/secrets.env (needed to pull weights on a cache miss)." >&2
  exit 1
fi
# SHARD_PSK gates the block-to-block ChaCha20 wire; default it if unset (the yaml does the same).
export SHARD_PSK="${SHARD_PSK:-cairn-fleet-psk}"
export CAIRN_API_KEY="$API_KEY"
export CAIRN_NWAY="$NWAY"
[ -n "$SELF_REPLENISH" ] && export CAIRN_SELF_REPLENISH="$SELF_REPLENISH"

if [ -z "$IMAGE_ID" ]; then
  echo "[bringup] ABORT: no Blackwell DLAMI known for region '$REGION'." >&2
  echo "[bringup] Add it to the DLAMI map above (or set IMAGE_ID=...). Re-resolve with:" >&2
  echo "  aws ec2 describe-images --owners amazon --region $REGION --filters \\" >&2
  echo "    \"Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20260619\"" >&2
  exit 1
fi

# ---------------------------------------------------------------------------------------------------
# 2. Spot sweep (read-only, no spend) — prices move a lot; eyeball the cheapest region before launch.
#    Informational only: we still launch in $REGION. Set DO_SWEEP=0 to skip.
# ---------------------------------------------------------------------------------------------------
if [ "$DO_SWEEP" = "1" ]; then
  echo "[bringup] spot sweep (read-only; informational — launching in \$REGION=$REGION regardless) ..."
  AWS_PROFILE="$PROFILE" python3 infra/skypilot/spot_sweep.py g7e.2xlarge --boxes "$NUM_NODES" || \
    echo "[bringup] (sweep failed/skipped — continuing)"
fi

# ---------------------------------------------------------------------------------------------------
# 3. Launch the fleet via the identity-safe wrapper (launch.sh restarts the shared sky api server under
#    cairn's profile so the box is tagged cairn=true with cairn's key — see launch.sh header). We pass
#    region + per-region AMI overrides, num_nodes, and every --env the dsv4 yaml reads. All the secret
#    --env names are bare (no =value) so sky forwards the value from THIS exported environment.
# ---------------------------------------------------------------------------------------------------
echo "[bringup] launching $CLUSTER ($NUM_NODES× g7e.2xlarge spot in $REGION) — detached ..."
CLUSTER="$CLUSTER" YAML="$YAML" CAIRN_IDLE_MIN="$IDLE_MIN" SKY="$SKY" \
  bash infra/skypilot/launch.sh \
    --region "$REGION" --image-id "$IMAGE_ID" --num-nodes "$NUM_NODES" \
    --env HF_TOKEN --env SHARD_PSK \
    --env CAIRN_API_KEY="$API_KEY" \
    --env CAIRN_NWAY="$NWAY" \
    --env CAIRN_S3_CACHE_ACCESS_KEY_ID --env CAIRN_S3_CACHE_SECRET_ACCESS_KEY \
    --env CAIRN_S3_WRITE_ACCESS_KEY_ID --env CAIRN_S3_WRITE_SECRET_ACCESS_KEY \
    --env CAIRN_ALERT_WEBHOOK --env CALLMEBOT_PHONE --env CALLMEBOT_APIKEY \
    ${SELF_REPLENISH:+--env CAIRN_SELF_REPLENISH="$SELF_REPLENISH" \
      --env CAIRN_LAUNCH_AWS_ACCESS_KEY_ID --env CAIRN_LAUNCH_AWS_SECRET_ACCESS_KEY} \
  || { echo "[bringup] launch failed — check the sky logs, then: infra/skypilot/nuke.sh (ground truth)"; exit 1; }

echo "[bringup] launch submitted. Setup (image + 294 GB weights + kernel) takes a while on a cold box."
echo "[bringup] follow it with:  AWS_PROFILE=$PROFILE $SKY logs $CLUSTER"

# ---------------------------------------------------------------------------------------------------
# 4. Wait for the rank0 serve_http endpoint to bind :8000. We probe over a short-lived ssh tunnel
#    (the box has no public :8000 — access is key-gated via ssh -L only). /v1/health is the readiness
#    signal serve_http exposes. Generous timeout because a COLD box pays image+weights+kernel first.
# ---------------------------------------------------------------------------------------------------
WAIT_SECS="${WAIT_SECS:-3600}"   # up to ~60 min for a fully-cold box (cache miss = HF weight pull)
PROBE_PORT="${PROBE_PORT:-18000}"
echo "[bringup] waiting for the endpoint to bind (up to ${WAIT_SECS}s; cold box pays setup first) ..."
deadline=$(( $(date +%s) + WAIT_SECS ))
bound=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  # open a throwaway tunnel, probe /v1/health, tear it down
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -fN -L "${PROBE_PORT}:localhost:8000" "$CLUSTER" 2>/dev/null
  TUN_PID=$!
  sleep 2
  if curl -fsS -m 5 "http://127.0.0.1:${PROBE_PORT}/v1/health" >/dev/null 2>&1; then
    bound=1
  fi
  kill "$TUN_PID" 2>/dev/null || true
  pkill -f "ssh.*-L ${PROBE_PORT}:localhost:8000" 2>/dev/null || true
  [ "$bound" = "1" ] && break
  sleep 20
done

if [ "$bound" != "1" ]; then
  echo "[bringup] endpoint did not bind within ${WAIT_SECS}s — it may still be in setup."
  echo "[bringup] check: AWS_PROFILE=$PROFILE $SKY logs $CLUSTER   (look for '[serve_http] Cairn endpoint on ...')"
  echo "[bringup] once up, tunnel manually:  ssh -L ${LOCAL_PORT}:localhost:8000 $CLUSTER -N &"
  # don't exit non-zero: the fleet IS launched; this is just the readiness probe timing out.
fi

# ---------------------------------------------------------------------------------------------------
# 5. Open the durable SSH tunnel to localhost:$LOCAL_PORT (the chat page + curl target).
# ---------------------------------------------------------------------------------------------------
if [ "$bound" = "1" ] && [ "$AUTO_TUNNEL" = "1" ]; then
  pkill -f "ssh.*-L ${LOCAL_PORT}:localhost:8000 $CLUSTER" 2>/dev/null || true
  ssh -o StrictHostKeyChecking=no -fN -L "${LOCAL_PORT}:localhost:8000" "$CLUSTER" \
    && echo "[bringup] tunnel up: localhost:${LOCAL_PORT} -> $CLUSTER:8000" \
    || echo "[bringup] tunnel failed; open it manually: ssh -L ${LOCAL_PORT}:localhost:8000 $CLUSTER -N &"
fi

# ---------------------------------------------------------------------------------------------------
# 6. How to chat + how to tear down.
# ---------------------------------------------------------------------------------------------------
cat <<EOF

============================================================================================
  Cairn fleet '$CLUSTER' is up — $NWAY active + $((NUM_NODES-NWAY)) warm spare(s), model=$MODEL
============================================================================================

  CHAT (built-in page):   open http://localhost:${LOCAL_PORT}/
  CHAT (local proxy UI):  python3 infra/skypilot/cairn-chat.py   # -> http://localhost:8001

  CURL:
    curl -s http://localhost:${LOCAL_PORT}/v1/chat/completions \\
      -H 'authorization: Bearer ${API_KEY}' -H 'content-type: application/json' \\
      -d '{"model":"${MODEL}","messages":[{"role":"user","content":"hi"}],"max_tokens":48}'

  TUNNEL (if not auto-opened):  ssh -L ${LOCAL_PORT}:localhost:8000 $CLUSTER -N &
  LOGS:                         AWS_PROFILE=$PROFILE $SKY logs $CLUSTER

  DRAIN-SENTINEL recovery test (migrate one rank to a warm spare, no kill):
    AWS_PROFILE=$PROFILE $SKY exec $CLUSTER "touch /tmp/cairn-shared/drain"

  TEARDOWN (EC2-API ground truth — works even when 'sky down' wedges):
    infra/skypilot/nuke.sh            # dry run: list every cairn=true box
    infra/skypilot/nuke.sh --force    # TERMINATE them all + verify via the EC2 API
    # (cairn=true is ground truth; 'sky status' can lie — see nuke.sh / the operator README)

============================================================================================
EOF
