# Cairn — Operator README

How to bring up, use, recover, and tear down a live Cairn endpoint.

Cairn serves a large open model (default **DeepSeek-V4-Flash FP8**, 671B MoE) by **pipeline-splitting it
by layers** across N cheap single-GPU AWS spot boxes (g7e / RTX PRO 6000 Blackwell, sm_120), and keeps
the endpoint alive through spot reclaims with a **warm-spare recovery layer** that can self-replenish.
The OpenAI-compatible BYOK endpoint (with a built-in chat page) is served by `cairn_node/serve_http.py`
on rank0 of the fleet.

> **The one thing to remember about teardown:** `cairn=true` (the EC2 tag) is ground truth, **not**
> `sky status`. Always verify with `infra/skypilot/nuke.sh`.

---

## 1. Prerequisites

| Need | What / where |
|------|--------------|
| **AWS scoped key** | The least-privilege `cairn-skypilot` IAM user, configured as the `cairn-skypilot` AWS CLI profile. It can only provision in Cairn's regions and only terminate `cairn=true`-tagged boxes. Never use root/admin keys. |
| **SkyPilot** | `pip install 'skypilot[aws]'`. `bringup.sh` calls the `sky` binary (auto-discovered, or set `SKY=/path/to/sky`). |
| **HF token** | A read-scoped HuggingFace token (`HF_TOKEN`) — pulls the model weights on a cache miss. |
| **`infra/secrets.env`** | Copy from the example and fill it in (next section). Gitignored; never committed. |
| **In-region weight/image cache (optional but recommended)** | `CAIRN_S3_CACHE_*` (read-only) syncs the 294 GB FP8 weights from in-region S3 instead of HF; `CAIRN_S3_WRITE_*` pulls the 82 GB sglang image from the in-region ECR mirror. Absent → graceful fallback to HF / Docker Hub. See [`infra/skypilot/CACHING.md`](../infra/skypilot/CACHING.md). |

### `infra/secrets.env`

```sh
cp infra/secrets.env.example infra/secrets.env
# then edit infra/secrets.env and fill in the values
```

The file is **bare `VAR=value`** lines (gitignored). Keys the dsv4 fleet reads:

```sh
HF_TOKEN=hf_xxxxxxxx                       # required (weight pull on cache miss)
SHARD_PSK=<openssl rand -hex 32>           # block-to-block ChaCha20 wire key (defaults if empty)
# In-region cache (optional — falls back to HF / Docker Hub if absent):
CAIRN_S3_CACHE_ACCESS_KEY_ID=...           # read-only key: weights from skypilot-cairn-weights-<region>
CAIRN_S3_CACHE_SECRET_ACCESS_KEY=...
CAIRN_S3_WRITE_ACCESS_KEY_ID=...           # scoped key: sglang image from the in-region ECR mirror
CAIRN_S3_WRITE_SECRET_ACCESS_KEY=...
# Spot-reclaim alert (optional — pick ONE channel; box-side drain works regardless):
CAIRN_ALERT_WEBHOOK=https://ntfy.sh/<topic>
# or CALLMEBOT_PHONE + CALLMEBOT_APIKEY (WhatsApp)
# Self-replenish launch key (only if you enable CAIRN_SELF_REPLENISH):
CAIRN_LAUNCH_AWS_ACCESS_KEY_ID=...
CAIRN_LAUNCH_AWS_SECRET_ACCESS_KEY=...
```

> **The `set -a` gotcha.** `source infra/secrets.env` alone does **not** export the vars, so the `sky`
> subprocess (and the `--env` passthroughs) can't see them. You must wrap it:
> `set -a; source infra/secrets.env; set +a`. `bringup.sh` does this for you — but if you ever load
> secrets by hand, remember the `set -a`.

---

## 2. Model-as-config

The served model is just an env var. The default is the live config:

```
MODEL=sgl-project/DeepSeek-V4-Flash-FP8
```

This single value is the served `model` name in `/v1/chat/completions`, the HF repo, and the
in-region S3 weight prefix. To serve a different model, set `MODEL=...` (and stage that model's weights
+ a matching kernel cache per `CACHING.md`). For the default dsv4 fleet, the FP8 checkpoint and the
sm_120 kernel are already wired.

---

## 3. The one-command bring-up

From the repo root:

```sh
cp infra/secrets.env.example infra/secrets.env   # first time only — then fill it in
bash infra/aws/ensure-fleet-sg.sh                # one-time per region: creates the shared `cairn-fleet` SG
bash infra/skypilot/bringup.sh
```

> **One-time per region:** the fleet and its spares all join a shared security group (`cairn-fleet`)
> so a replenished spare — which launches as a *separate* SkyPilot cluster — can still reach the
> recovery wire (ports 7777–7780). `ensure-fleet-sg.sh` creates that SG with the right rules; skip it
> and recovery onto a replenished spare will silently fail. (See `report/productization.md`.)

That's it. `bringup.sh`:

1. **Loads secrets** correctly (`set -a; source infra/secrets.env; set +a`).
2. **Spot-sweeps** (read-only, no spend) so you can eyeball the cheapest region.
3. **Launches the fleet** via the identity-safe `launch.sh` wrapper (which restarts the shared sky api
   server under cairn's profile so boxes are tagged `cairn=true` with cairn's key).
4. **Waits** for the rank0 `serve_http` endpoint to bind `:8000` (probes `/v1/health` over a tunnel).
5. **Opens the SSH tunnel** `localhost:8000 -> rank0:8000`.
6. **Prints** how to chat and how to tear down.

### Defaults (match the current live config)

| Knob (env var) | Default | Meaning |
|----------------|---------|---------|
| `MODEL` | `sgl-project/DeepSeek-V4-Flash-FP8` | served model / weight id |
| `NWAY` | `4` | **active** layer-split stages (FP8 needs 4; 3-way OOMs) |
| `NUM_NODES` | `6` | total boxes = `NWAY` active + `(NUM_NODES-NWAY)` warm spares → **4 active + 2 spares** |
| `REGION` | `us-east-2` | Ohio — cheapest g7e spot (2026-06-24) |
| `WARM_TARGET` | `NUM_NODES-NWAY` | warm spares to keep |
| `CLUSTER` | `cairn-dsv4` | sky cluster name (also the `ssh <name>` host alias) |
| `LOCAL_PORT` | `8000` | local port the tunnel binds |
| `CAIRN_API_KEY` | `sk-cairn-demo` | BYOK Bearer key the endpoint enforces |
| `CAIRN_SELF_REPLENISH` | (off) | non-empty → provision a replacement spare on consumption |

Override any inline, e.g. a 3-way fleet in Spain with 5 boxes:

```sh
NWAY=3 NUM_NODES=5 REGION=eu-south-2 bash infra/skypilot/bringup.sh
```

> **Cold-box note.** A first launch in a region with an empty cache pulls 294 GB of weights from HF and
> may build the kernel — setup can take ~30–60 min before the endpoint binds. With the in-region S3
> cache populated it's an in-region sync (much faster). The readiness probe waits up to `WAIT_SECS`
> (default 3600s); the fleet is up regardless of whether the probe times out.

---

## 4. What each knob does

### N-way topology (`NWAY`)
The model's 43 layers are split across `NWAY` **active** stages, each on its own GPU box:
- `NWAY=4` → layer counts `11,11,11,10` (the resilience topology; FP8 fits at 4-way).
- `NWAY=3` → `15,14,14` (OOMs on FP8 — only for smaller checkpoints).

rank0 = ENTRY (+ the `serve_http` endpoint, co-located on `:8000`), middle ranks forward stage→stage,
the last active rank = TAIL. The split is pinned via `CAIRN_PP_LAYER_PARTITION` so sglang slices to
exactly these boundaries.

### Warm pool / warm-target (`WARM_TARGET`, spares = `NUM_NODES - NWAY`)
Boxes with rank `>= NWAY` are **generic warm spares**: weights loaded + flashinfer pre-warmed at launch,
sitting idle, dialing the driver's spare-sink and announcing their address. When a stage dies, the
driver re-stitches the in-flight stream onto a warm spare — pre-warmed, so the swap is ~tens of ms, not
a cold start. `serve_http` keeps the pool topped up to `--warm-target` (the yaml derives
`--warm-target`/`--initial-spares` from `num_nodes - NWAY`).

### Partition / load-on-promotion (`--stage-partition`)
The driver knows each active rank's layer counts, so a **generic** spare can cover **any** position: on
promotion it re-execs to load slice `k` from the local NVMe instance store (every box holds the full
294 GB checkpoint on NVMe; only its slice is resident in VRAM). This is why a single warm pool covers
entry/middle/tail, not just the tail.

### Self-replenish (`CAIRN_SELF_REPLENISH`)
Off by default (needs an admin-provisioned scoped launch key on the box, `CAIRN_LAUNCH_AWS_*`). When
on: after a spare is **consumed** by a recovery, `serve_http` writes a request to `/shared/...`, and a
**host-side watcher** (`infra/skypilot/replenish-watcher.sh`, outside the serving container — AWS creds
stay off the container) launches a fresh replacement spare (`cairn-dsv4-spare.sky.yaml`) that dials
back + announces, refilling the pool toward `warm_target`. Capped by `CAIRN_REPLENISH_MAX` (default 10).

---

## 5. Using the endpoint

The box has **no public `:8000`** — access is key-gated over the SSH tunnel only (`bringup.sh` opens
it; if needed manually: `ssh -L 8000:localhost:8000 cairn-dsv4 -N &`).

### Built-in chat page
`serve_http` serves a self-contained chat UI at `GET /` (same-origin, no CORS, key injected server-side):

```sh
open http://localhost:8000/
```

### curl

```sh
curl -s http://localhost:8000/v1/chat/completions \
  -H 'authorization: Bearer sk-cairn-demo' -H 'content-type: application/json' \
  -d '{"model":"sgl-project/DeepSeek-V4-Flash-FP8",
       "messages":[{"role":"user","content":"hi"}],"max_tokens":48}'
```

`401` without the key, `200` with it. Responses carry a `cairn` field; if a recovery happened
mid-request it includes `{"recovered": true, "mttr_s": ...}`.

### Local proxy chat (`cairn-chat.py`)
If you're running a `serve_http` build without the built-in page (or want a separate UI), this is a tiny
local proxy that serves a chat page on `:8001` and forwards `/v1/*` to the tunneled endpoint:

```sh
ssh -L 8000:localhost:8000 cairn-dsv4 -N &       # tunnel first
python3 infra/skypilot/cairn-chat.py             # -> http://localhost:8001
```

Env: `CAIRN_UPSTREAM` (default `http://127.0.0.1:8000`), `CAIRN_API_KEY`, `CAIRN_MODEL`, `CAIRN_CHAT_PORT`.

---

## 6. Recovery behavior & the drain-sentinel test hook

**How recovery works.** A spot reclaim is ~2-min pre-warned via IMDS. The **primary path is proactive
drain-before-death**: the doomed box signals the driver while still alive, the driver migrates the
in-flight stream to a warm spare bit-identically (no dropped token), and the box exits cleanly. The
**fallback is reactive** (after-death half-open timeout → re-stitch → replay committed history →
resume). Either way the endpoint keeps serving. (Live-proven: bit-identical migration, pre-warmed swap
~23 ms.)

**Drain-sentinel test hook.** Every rank arms a sentinel watcher on `CAIRN_SPOT_TEST_FILE=/shared/drain`
(the host `/tmp/cairn-shared` is bind-mounted to the container's `/shared`). Touch it to drain **that**
box's rank and force a graceful migration to a warm spare — no real reclaim needed:

```sh
AWS_PROFILE=cairn-skypilot sky exec cairn-dsv4 "touch /tmp/cairn-shared/drain"
```

Watch the endpoint stay up across the migration (the response's `cairn.recovered` flag confirms it).
An abrupt-kill variant (exercises the reactive path) is in `cairn-dsv4.sky.yaml`'s header:
`sky exec cairn-dsv4 "pkill -9 -f 'layer-start 33'"`.

---

## 7. Teardown + EC2-API verification

**`cairn=true` (the EC2 tag) is ground truth — `sky status` can lie.** `sky down` cannot terminate an
INIT-wedged cluster (it hangs while the boxes keep billing). The kill switch talks to the EC2 API
directly and only terminates `cairn=true` boxes (the scoped key can't touch anything else):

```sh
infra/skypilot/nuke.sh            # DRY RUN — list every cairn=true box across Cairn's regions
infra/skypilot/nuke.sh --force    # TERMINATE them all, then re-print to confirm
```

After `--force`, re-run the dry-run form until it prints **"No cairn instances … nothing billing"** —
that, not `sky status`, is your confirmation nothing is billing. `nuke.sh` sweeps
`eu-south-2 ap-northeast-2` by default; override with `CAIRN_REGIONS="us-east-2 us-west-2 ..."` to cover
the region you launched in.

The graceful path (`AWS_PROFILE=cairn-skypilot sky down cairn-dsv4`) is fine when the cluster is
healthy — but **always** finish with the `nuke.sh` ground-truth check.

---

## 8. Cheat sheet

```sh
# Bring up (defaults: dsv4 FP8, 4 active + 2 spares, us-east-2):
bash infra/skypilot/bringup.sh

# Chat:
open http://localhost:8000/

# Logs:
AWS_PROFILE=cairn-skypilot sky logs cairn-dsv4

# Drain-sentinel recovery test:
AWS_PROFILE=cairn-skypilot sky exec cairn-dsv4 "touch /tmp/cairn-shared/drain"

# Tear down (ground truth):
infra/skypilot/nuke.sh --force
```
