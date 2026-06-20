#!/usr/bin/env python3
"""Launch the Cairn spot fleet — entirely via the SkyPilot Python API, no AWS console.

    python infra/skypilot/launch.py --model gpt-oss-120b            # N (from the fit) + 1 spare
    python infra/skypilot/launch.py --model gpt-oss-120b --spares 2 --dry-run

What it does:
  1. Computes N from the model fit (the same `cairn_scheduler.fit` the scheduler uses).
  2. Provisions N + `spares` single-GPU g6.xlarge SPOT instances in one VPC/placement
     group (cairn-block.sky.yaml), each as a SkyPilot **managed job** so SkyPilot
     auto-recovers a preempted instance (the minutes-timescale backfill, handoff §4).
  3. Injects secrets (SHARD_PSK, HF_TOKEN, CAIRN_CONTROL_URL) from the environment /
     secret store at launch — never from the repo (spec §8).

⚠️ GPU/AWS-UNVERIFIED. The SkyPilot primitive choice (managed jobs vs cluster) is the
seam the handoff says to validate live (handoff §3) — confirm on the pinned SkyPilot
version before relying on it. Structure + the API calls are here; run it with AWS creds.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scheduler" / "src"))

from cairn_scheduler import fit, load_model_config  # noqa: E402

BLOCK_TASK = ROOT / "infra" / "skypilot" / "cairn-block.sky.yaml"
REQUIRED_SECRETS = ("SHARD_PSK", "HF_TOKEN", "CAIRN_CONTROL_URL")


def _load_secrets_file() -> None:
    """Auto-load infra/secrets.env (gitignored) into the environment, so HF_TOKEN / SHARD_PSK /
    CAIRN_CONTROL_URL transfer to every fleet node at launch without hand-copying. Explicit env
    vars win (setdefault). The file is NEVER committed (.gitignore); the repo ships only .example."""
    f = ROOT / "infra" / "secrets.env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _secrets() -> dict:
    _load_secrets_file()
    missing = [k for k in REQUIRED_SECRETS if not os.environ.get(k)]
    if missing:
        raise SystemExit(
            f"[launch] missing secrets: {', '.join(missing)}. Put them in infra/secrets.env "
            f"(cp infra/secrets.env.example) or export them — never the repo (spec §8)."
        )
    return {k: os.environ[k] for k in REQUIRED_SECRETS}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-oss-120b")
    ap.add_argument("--spares", type=int, default=1, help="warm spares (spec §5.2 v1.0 default: 1)")
    ap.add_argument("--region", default=None, help="override the model's pool region (default: from configs/<model>.yaml)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_model_config(ROOT / "configs" / f"{args.model}.yaml")
    n = fit(cfg).n
    instance_type = cfg.instance_type   # pool is model-config (inv #3): g6.xlarge proof / g7e.2xlarge GLM-5.2 headline
    region = args.region or cfg.region  # region is model-config too: proof=eu-south-2 (Spain) / GLM headline=ap-northeast-2 (Seoul)
    total = n + args.spares
    print(f"[launch] {args.model}: N={n} block nodes + {args.spares} warm spare(s) = {total} "
          f"{instance_type} spot in {region} (one VPC/placement group)")

    if args.dry_run:
        print("[launch] dry-run: no resources provisioned.")
        return 0

    secrets = _secrets()
    try:
        import sky  # heavy (cloud SDKs) — imported only on a real launch
    except ImportError:
        raise SystemExit("[launch] SkyPilot not installed. `pip install 'skypilot[aws]'` (infra/).")

    for i in range(total):
        role = "spare" if i >= n else f"block-{i}"
        task = sky.Task.from_yaml(str(BLOCK_TASK))
        # Pool is model-config (inv #3): override the template's instance_type per model
        # (g6.xlarge proof / g7e.2xlarge GLM-5.2 headline) + region. Validate the exact
        # override field on the pinned SkyPilot version at first live launch (handoff §3).
        base_res = list(task.resources)[0]
        task.set_resources(base_res.copy(instance_type=instance_type, region=region))
        task.update_envs({**secrets, "CAIRN_NODE_ROLE": role})
        # Managed job per node → SkyPilot auto-recovers spot preemptions (handoff §4).
        # NOTE(handoff §3): validate this primitive vs a multi-node cluster on the pinned
        # SkyPilot version before committing — heterogeneous-block pipeline != SkyServe replicas.
        sky.jobs.launch(task, name=f"cairn-{role}")
        print(f"[launch] submitted {role}")

    print(f"[launch] fleet up. Tear down with: python infra/skypilot/down.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
