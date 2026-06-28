# infra/ — AWS data-plane: launch & teardown strategy (API-only, no console)

The spot fleet is provisioned, recovered, and torn down **entirely through SkyPilot's
Python/CLI API + the AWS CLI**. Nobody clicks around the AWS web console — the only
human step is running `bootstrap.sh` once with admin creds; everything after is scripted.

> The SkyPilot primitive choice (managed jobs vs cluster) is a seam to validate live —
> confirm on the pinned SkyPilot version before relying on it. The live DeepSeek-V4-Flash
> run uses `cairn-dsv4.sky.yaml` directly (single SkyPilot cluster); `launch.py` /
> `cairn-block.sky.yaml` are the managed-job variant.

## Topology

- **One region, one VPC, one placement group** — low intra-pipeline RTT; the decode loop
  is wholly in-VPC. Multi-AZ is a spot-supply hedge, never mid-request.
- **N single-GPU `g7e.2xlarge` spot** instances (1× RTX PRO 6000 Blackwell, sm_120, 96 GB)
  + **1 warm spare**. N comes from the fit (`cairn_scheduler.fit`; DeepSeek-V4-Flash FP8,
  ~294 GB → N=4). Spot-only, **0 on-demand floor**.
- Each node serves one contiguous block of layers (`cairn-block.sky.yaml`). Cairn's
  controller assigns the block + pipeline position and stitches stages over the wire —
  **not** SkyPilot, **not** SGLang's NCCL PP (which can't survive a node vanishing).

## The lifecycle — every step is an API call

```sh
# 0. One-time, via AWS CLI (API, not console): IAM user + least-privilege policy + keys
bash infra/aws/bootstrap.sh                      # uses infra/aws/iam-policy.json

# 1. Secrets in the environment / secret store — never the repo
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export SHARD_PSK=$(openssl rand -hex 32)  HF_TOKEN=...  CAIRN_CONTROL_URL=https://<worker>/...
pip install 'skypilot[aws]' && sky check aws

# 2. Launch the fleet (N + spare), fully scripted
python infra/skypilot/launch.py --model deepseek-v4-flash-fp8   # add --dry-run to preview N

# 3. Tear down → scale to zero (no idle GPU spend)
python infra/skypilot/down.py
```

## Spot recovery — two timescales

1. **Seconds (Cairn):** a node is reclaimed → the controller reassigns its block to the
   **warm spare** (VRAM-load + CUDA-graph capture, no download), replays the
   KV from the durable token-history, re-stitches the edges, resumes. Serving never stops
   for the streams not through the dead stage.
2. **Minutes (SkyPilot):** SkyPilot relaunches a replacement spot instance (managed-job
   auto-recovery) → it becomes the **new** warm spare, restoring the safety margin.

Cairn owns half 1; SkyPilot owns half 2. They run at their own timescales and are built
independently.

## Cost discipline

- **Spot-only**, **scale-to-zero** on teardown → you pay for GPUs only while serving.
- The only standing waste while up is the **1 warm spare**; at rest, zero.
- The economic thesis is *supply arbitrage* (cheap deep single-GPU spot pools), eroded by
  the spare + any rebuild thrash — watch `cost/token` and the thrash bound.

## No-console guarantee

| Step | How | Console? |
|---|---|---|
| IAM user + policy + keys | `aws iam …` (`bootstrap.sh`) | ❌ API |
| Provision N + spare spot | `sky.jobs.launch` (`launch.py`) | ❌ API |
| Spot recovery | SkyPilot managed-job auto-recovery | ❌ automatic |
| Teardown / scale-to-zero | `sky.jobs.cancel` (`down.py`) | ❌ API |

## Files

- `skypilot/cairn-dsv4.sky.yaml` — the live DeepSeek-V4-Flash FP8 fleet (g7e Blackwell spot,
  4-way layer split + warm spares, single SkyPilot cluster).
- `skypilot/cairn-block.sky.yaml` — one block node (managed-job variant, setup + serve).
- `skypilot/launch.py` — provision N (from the fit) + spares; inject secrets.
- `skypilot/down.py` — cancel all `cairn-*` jobs (scale to zero).
- `skypilot/measure.py` — on-box probe (usable VRAM + framework/activation overhead → config values).
- `requirements-gpu.txt` — the pinned box image (torch CUDA + SGLang + the fork; transformers comes via sglang).
- `requirements-ref.txt` — the CPU reference-oracle env (transformers 5.x), a SEPARATE venv from the box image.
- `aws/iam-policy.json` + `aws/bootstrap.sh` — least-privilege IAM, created via the CLI.

**Open (validate live):** the SkyPilot primitive (managed jobs vs cluster) for a
heterogeneous-block pipeline; the exact placement-group wiring; the pinned SkyPilot +
SGLang + torch-CUDA versions.
