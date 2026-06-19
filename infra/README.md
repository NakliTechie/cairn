# infra/ — AWS data-plane: launch & teardown strategy (API-only, no console)

The spot fleet is provisioned, recovered, and torn down **entirely through SkyPilot's
Python/CLI API + the AWS CLI**. Nobody clicks around the AWS web console — the only
human step is running `bootstrap.sh` once with admin creds; everything after is scripted.

> ⚠️ GPU/AWS-unverified. These configs + scripts are the rung-2/3 plan, structured and
> ready to run on a GPU pool with AWS creds. The SkyPilot primitive choice (managed jobs
> vs cluster) is the seam the handoff says to validate live (handoff §3) — confirm on the
> pinned SkyPilot version before relying on it.

## Topology (v1.0)

- **One region, one VPC, one placement group** — low intra-pipeline RTT; the decode loop
  is wholly in-VPC (invariant #4). Multi-AZ is a v1.2 spot-supply hedge, never mid-request.
- **N single-GPU `g6.xlarge` spot** instances (1× L4, 24 GB) + **1 warm spare**. N comes
  from the fit (`cairn_scheduler.fit`; gpt-oss-120b → N≈4). Spot-only, **0 on-demand floor**
  (spec §5.2 v1.0 default).
- Each node serves one contiguous block of layers (`cairn-block.sky.yaml`). Cairn's
  controller assigns the block + pipeline position and stitches stages over the wire —
  **not** SkyPilot, **not** SGLang's NCCL PP (which can't survive a node vanishing).

## The lifecycle — every step is an API call

```sh
# 0. One-time, via AWS CLI (API, not console): IAM user + least-privilege policy + keys
bash infra/aws/bootstrap.sh                      # uses infra/aws/iam-policy.json

# 1. Secrets in the environment / secret store — never the repo (spec §8)
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...
export SHARD_PSK=$(openssl rand -hex 32)  HF_TOKEN=...  CAIRN_CONTROL_URL=https://<worker>/...
pip install 'skypilot[aws]' && sky check aws

# 2. Launch the fleet (N + spare), fully scripted
python infra/skypilot/launch.py --model gpt-oss-120b      # add --dry-run to preview N

# 3. Tear down → scale to zero (no idle GPU spend)
python infra/skypilot/down.py
```

## Spot recovery — two timescales (handoff §4)

1. **Seconds (Cairn):** a node is reclaimed → the controller reassigns its block to the
   **warm spare** (VRAM-load + CUDA-graph capture, no download — spec §5.1), replays the
   KV from the durable token-history, re-stitches the edges, resumes. Serving never stops
   for the streams not through the dead stage.
2. **Minutes (SkyPilot):** SkyPilot relaunches a replacement spot instance (managed-job
   auto-recovery) → it becomes the **new** warm spare, restoring the safety margin.

Cairn owns half 1; SkyPilot owns half 2. They run at their own timescales and are built
independently.

## Cost discipline (vision §6)

- **Spot-only**, **scale-to-zero** on teardown → you pay for GPUs only while serving.
- The only standing waste while up is the **1 warm spare** (the §5.2 floor); at rest, zero.
- The economic thesis is *supply arbitrage* (cheap deep single-GPU spot pools), eroded by
  the spare + any rebuild thrash — watch `cost/token` and the thrash bound (spec §10).

## No-console guarantee

| Step | How | Console? |
|---|---|---|
| IAM user + policy + keys | `aws iam …` (`bootstrap.sh`) | ❌ API |
| Provision N + spare spot | `sky.jobs.launch` (`launch.py`) | ❌ API |
| Spot recovery | SkyPilot managed-job auto-recovery | ❌ automatic |
| Teardown / scale-to-zero | `sky.jobs.cancel` (`down.py`) | ❌ API |

## Files

- `skypilot/cairn-block.sky.yaml` — one block node (g6.xlarge spot, setup + serve).
- `skypilot/launch.py` — provision N (from the fit) + spares; inject secrets.
- `skypilot/down.py` — cancel all `cairn-*` jobs (scale to zero).
- `requirements-gpu.txt` — the pinned box image (torch CUDA + SGLang + the fork).
- `aws/iam-policy.json` + `aws/bootstrap.sh` — least-privilege IAM, created via the CLI.

**Open (validate live, handoff §3):** the SkyPilot primitive (managed jobs vs cluster) for
a heterogeneous-block pipeline; the exact placement-group wiring; the pinned SkyPilot +
SGLang + torch-CUDA versions.
