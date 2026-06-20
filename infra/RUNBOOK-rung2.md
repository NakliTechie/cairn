# Cairn — AWS bring-up & rung-2 GPU runbook

Executes workplan **Chunk A → B → C**: wire the AWS data plane end-to-end and stand up
the first real GPU serving. **The GPU *is* the AWS spot box**, region is **per-model**:
`g6.xlarge`/L4 for the gpt-oss-120b proof in **`eu-south-2` (Spain, <5% interruption)**, and
`g7e.2xlarge`/RTX PRO 6000 Blackwell for the **GLM-5.2 (NVFP4) headline in `ap-northeast-2`
(Seoul)** where live g7e spot is ~40% cheaper. Set in `configs/<model>.yaml`. Everything is API/CLI-driven.

> Status when this was written: rung-1 sim + fork + CF control plane are built, tested
> (~110 tests), and pushed. The `BlockRuntime` seam, the wire (crypto-verified), the LAN
> transport, the adapter, the scheduler/recovery, and the SkyPilot configs are all ready —
> only the SGLang block-forward and the live AWS run are left.

## Objectives — what this runbook builds toward
Full statement: [`docs/cairn-objectives.md`](../docs/cairn-objectives.md). End state:
1. A **working system** proven up the cost-ladder (rungs 0→5).
2. **Benchmark metrics** (Chunk C gate artifacts) — cost/token vs on-demand, MTTR, occupancy, split-correctness, induced-interruption — **head-to-head vs SpotServe (54%) + KevlarFlow (MTTR) on their exact models** (gpt-neox-20b, llama-3.1-8b).
3. A **paper** (GitHub / arXiv).
4. An **open-source repo** people run in production.

The honest contribution = a **minimal hot-swap failover layer** for pipeline LLM serving on spot: on-demand-grade reliability at ~spot cost. Open source; consulting upside, not a product.

## What the user provides (the only blockers)
1. **AWS account access** — admin once: enable the **`eu-south-2` (Spain) opt-in region**, mint
   the scoped key, and file the **G-spot vCPU quota** (Step 2.5 — has review lead time, start early).
2. **HF token** — to pull model weights (gpt-oss-120b proof; GLM-5.2-NVFP4 headline; a 7–9B for rung-2 bring-up).
3. **Acceptance of spot spend** — proof ~$0.8/hr, GLM-5.2 headline ~$2.3/hr while a fleet is up; `down.py` scales to zero.

Secrets handling: put them in **`infra/secrets.env`** (gitignored — `cp infra/secrets.env.example`
and fill `HF_TOKEN` / `SHARD_PSK` / `CAIRN_CONTROL_URL`). `launch.py` auto-loads it and injects to
every node at boot, so the HF token transfers to the fleet without hand-copying. Never the repo
(`.gitignore` blocks `secrets.env`/`*.key`/`*.pem`/`.env`). I never echo or commit a key.

---

## Step 1 — Mint the scoped IAM key  (Chunk A)
The least-privilege policy already exists: [`aws/iam-policy.json`](aws/iam-policy.json)
(region-locked, destructive actions tag-gated, `PassRole` pinned to EC2).

- **Option A — Claude-in-Chrome MCP (you're console-unfamiliar):** you log into the AWS
  console in Chrome; I drive IAM → Users → create `cairn-skypilot` → attach the inline policy
  (paste `iam-policy.json` minus the `_comment`) → Security credentials → create access key →
  you hand me the key id + secret (I put them in env, never the repo).
- **Option B — CLI:** with admin creds in your shell, run `bash aws/bootstrap.sh` — it does the
  whole thing via `aws iam` and prints the key once.

**Done when:** `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` are in my env and `aws sts
get-caller-identity` returns `cairn-skypilot`.

## Step 2 — Validate the Batch-D hardening live  (Chunk A — the owed `[test]`s)
```sh
pip install 'skypilot[aws]' && sky check aws            # confirms the key works for provisioning
aws iam simulate-principal-policy ...                   # H5/H6: prove the scope holds + nothing over-broad
```
- If SkyPilot fails a specific action, **loosen that one action** (cross-check SkyPilot's
  documented policy) — never revert to `Resource:"*"`.
- Confirm the SG ingress for **:7777** is restricted to the fleet SG/CIDR (M13), not `0.0.0.0/0`.

**Done when:** `sky check aws` is green, the policy-sim shows least-privilege intact, SG is tight.

## Step 2.5 — Spot quota (start EARLY — has review lead time)  (Chunk A)
G-family spot defaults to **0 vCPU**, and new GPU families hit a human "graded" review (partial
grant first, then Spot auto-ramps with demonstrated usage). One quota covers both pools:
**"All G and VT Spot Instance Requests"** (`L-3819A6DF`, vCPU-based, per-region). Do this with the
**admin** login, not the scoped key. The region must be **enabled first** (Spain is opt-in).
```sh
aws account enable-region --region-name eu-south-2          # opt-in; wait for ENABLED (mins–hours)
# need: proof ~20 vCPU (g6.xlarge=4) + GLM-5.2 ~48–56 vCPU (g7e.2xlarge=8). Ask conservatively, then ramp:
aws service-quotas request-service-quota-increase \
  --service-code ec2 --quota-code L-3819A6DF --desired-value 64 --region eu-south-2
aws service-quotas get-service-quota --service-code ec2 --quota-code L-3819A6DF --region eu-south-2  # applied ≥ need?
```
- **Justify** (if routed to a case): fault-tolerant LLM serving; spot because the workload is
  interruption-tolerant *by design* (Cairn recovers). Lead with that — good-spot-citizens clear faster.
- **Spot ≠ on-demand** (`L-DB2E81BA` is the on-demand G quota — request a little only for a non-spot bring-up box).
- **Spot Placement Score** ("6× g7e in one Spain AZ?") needs the region enabled but **no quota** — run it first to de-risk the AZ.

**Done when:** the applied G-spot vCPU value in eu-south-2 covers the fleet you're about to launch.

## Step 3 — Implement the SGLang block-forward  (Chunk B, the net-new GPU work)
On a single GPU box first (cheapest iteration):
```sh
pip install -r requirements-gpu.txt                     # pin the torch CUDA wheel to the L4 driver
pip-compile --generate-hashes -o requirements.lock requirements-gpu.txt   # M12 hash-lock, on the box
```
Fill the two marked integration points in [`../fork/shard/sglang_node.py`](../fork/shard/sglang_node.py):
- `load_shard()` — load only `[layer_start, layer_end)` to VRAM + CUDA-graph capture.
- `forward(hidden, kv_meta)` — run the block over `hidden`, per-seq paged KV for `kv_meta["seq"]`.
- **Cheapest first step (path C):** a transformers reference block forward (no SGLang) to prove
  split-correctness + KV-replay, *then* swap to the SGLang path for performance.
- While here: measure the **§12 numbers** on the real L4 (usable VRAM, framework/activation
  overhead) and replace the placeholders in `configs/gpt-oss-120b.yaml`.

**Done when:** a single `SglangNodeRuntime` loads a block and `forward` returns the right shape.

## Step 4 — 2-GPU small-model run  (Chunk B gate — handoff §6 rung 2)
Split a small 7–9B across 2 GPUs (one box w/ 2 cards, or 2 spot nodes on the LAN):
```python
from adapter import build_shard_pipeline
pipeline = build_shard_pipeline(fit(cfg), model, runtime_cls=SglangNodeRuntime)
# drive it with the SAME cairn_scheduler.Scheduler used in the sim
```
Prove on **real tensors**: split output == single-GPU reference (greedy, token-for-token);
kill a node mid-decode → reassign → replay-rebuild KV → resume, output uncorrupted; the wire
carries activations. These are the rung-1 sim gates, now real.

**Done when:** the three gates pass on 2 real GPUs.

## Step 5 — Real-fleet v1.0 / v1.1 gate artifacts  (Chunk C)
```sh
python skypilot/launch.py --model gpt-oss-120b          # PROOF: N=4 g6.xlarge spot + 1 spare — eu-south-2 (Spain)
# ⚠ BEFORE the headline: widen the LIVE cairn-skypilot IAM policy to add ap-northeast-2 — it was minted eu-south-2-only.
python skypilot/launch.py --model glm-5.2               # HEADLINE: N=6 g7e.2xlarge (Blackwell, NVFP4) — ap-northeast-2 (Seoul, ~40% cheaper)
# ... run the bench harness against the live fleet → commit artifacts under bench/ ...
python skypilot/down.py                                 # scale to zero — no idle spend
```
- **v1.0:** reliability log, split-correctness, induced-interruption timeline, crypto self-test.
- **v1.1:** per-stage occupancy ≥ floor, cost/token vs the on-demand baseline (AWS g7e on-demand, same hardware), cross-stream correctness.
- Set the gate **thresholds** (X / Y / occupancy floor / baseline) from the §12 measurements (open question).

**Done when:** the v1.0 + v1.1 gate artifacts are committed under `bench/` from a real run.

---

## Gotchas (carried from the design + handoff)
- **Validate the SkyPilot primitive** (managed-jobs-per-node vs cluster) on the first live launch —
  the handoff flags this as "do not assume" (§3). Loosen IAM per-action if needed.
- **Two-timescale recovery:** the warm spare covers the ~2-min eviction gap in *seconds* (Cairn);
  SkyPilot backfills the instance in *minutes*. You cannot cold-start a replacement inside the
  eviction window — spares are pre-staged (disk-warm, VRAM-cold). This is the make-or-break loop.
- **No surprise weight downloads** — pulling gpt-oss-120b (~63 GB) is an explicit `cairn_node.stage`
  step, pre-staged to EBS so a reassign is VRAM-load only.
- **Cost:** spot-only, 0 on-demand floor (v1.0); the only standing waste is the 1 warm spare;
  `down.py` → zero at rest. Watch `cost/token` + the thrash bound (spec §10) — the economic truth.

## How I'll work it tomorrow
Point me at AWS (Chrome logged in, or admin creds in the shell) + the HF token. I'll drive
Steps 1–2 immediately (key + live validation), then 3–4 on a single/2-GPU box, then 5 on the
g6 pool. I verify at each "Done when" before moving on, and tear down between runs.
