# Cairn — AWS bring-up & rung-2 GPU runbook

Executes workplan **Chunk A → B → C**: wire the AWS data plane end-to-end and stand up
the first real GPU serving. **The GPU *is* the AWS spot box** (g6.xlarge / L4), so there is
one thread here, not two. Everything is API/CLI-driven — no AWS web-console clicking.

> Status when this was written: rung-1 sim + fork + CF control plane are built, tested
> (~110 tests), and pushed. The `BlockRuntime` seam, the wire (crypto-verified), the LAN
> transport, the adapter, the scheduler/recovery, and the SkyPilot configs are all ready —
> only the SGLang block-forward and the live AWS run are left.

## What the user provides (the only blockers)
1. **AWS account access** — admin once (to mint the scoped key), then nothing.
2. **HF token** — to pull model weights (gpt-oss-120b; a 7–9B for the rung-2 proof).
3. **Acceptance of g6 spot spend** — a few dollars/hour while a fleet is up; `down.py` scales to zero.

Secrets handling: everything goes in **env vars / a secret store**, never the repo
(`.gitignore` blocks `*.key`/`*.pem`/`.env`). I never echo or commit a key.

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
python skypilot/launch.py --model gpt-oss-120b          # N≈4 g6.xlarge spot + 1 warm spare, one VPC/PG
# ... run the bench harness against the live fleet → commit artifacts under bench/ ...
python skypilot/down.py                                 # scale to zero — no idle spend
```
- **v1.0:** reliability log, split-correctness, induced-interruption timeline, crypto self-test.
- **v1.1:** per-stage occupancy ≥ floor, cost/token vs the on-demand baseline, cross-stream correctness.
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
