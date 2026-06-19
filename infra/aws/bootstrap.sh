#!/usr/bin/env bash
# One-time AWS setup for Cairn — via the AWS CLI (API), NOT the web console.
# Creates a dedicated 'cairn-skypilot' IAM user, attaches the least-privilege policy, and
# mints an access key for SkyPilot. Run once with admin creds; thereafter the fleet is
# launched/torn-down entirely by infra/skypilot/{launch,down}.py.
#
#   aws sts get-caller-identity                 # confirm you're admin first
#   bash infra/aws/bootstrap.sh
#
# Output: an access key id + secret. Put them in your secret store / ~/.aws (never the repo).
set -euo pipefail

USER=cairn-skypilot
POLICY_FILE="$(dirname "$0")/iam-policy.json"

echo "[bootstrap] creating IAM user $USER (idempotent)"
aws iam get-user --user-name "$USER" >/dev/null 2>&1 || aws iam create-user --user-name "$USER"

echo "[bootstrap] attaching inline least-privilege policy"
# strip the _comment field the AWS API won't accept
TMP="$(mktemp)"; python3 -c "import json,sys;d=json.load(open('$POLICY_FILE'));d.pop('_comment',None);json.dump(d,open('$TMP','w'))"
aws iam put-user-policy --user-name "$USER" --policy-name cairn-skypilot --policy-document "file://$TMP"
rm -f "$TMP"

echo "[bootstrap] minting access key (store these in your secret store, NOT the repo):"
aws iam create-access-key --user-name "$USER" --output json

cat <<'EOF'

[bootstrap] done — all via API, no console. Next:
  export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...   # from above
  export SHARD_PSK=$(openssl rand -hex 32)
  export HF_TOKEN=...            CAIRN_CONTROL_URL=https://<your-worker>/...
  pip install 'skypilot[aws]' && sky check aws
  python infra/skypilot/launch.py --model gpt-oss-120b
EOF
