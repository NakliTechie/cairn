#!/usr/bin/env python3
"""Tear the Cairn fleet down — entirely via the SkyPilot API, no AWS console.

    python infra/skypilot/down.py              # cancel all cairn-* managed jobs (scale to zero)
    python infra/skypilot/down.py --list       # just show what's running

Cost discipline: spot-only + clean teardown means no idle GPU spend. The only standing
cost while up is the warm spare; at rest, zero. There is no console step — this is the
down half of a fully-API lifecycle.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--prefix", default="cairn-")
    args = ap.parse_args()

    try:
        import sky
    except ImportError:
        raise SystemExit("[down] SkyPilot not installed. `pip install 'skypilot[aws]'`.")

    jobs = [j for j in sky.jobs.queue(refresh=True) if j.get("job_name", "").startswith(args.prefix)]
    if args.list:
        for j in jobs:
            print(f"  {j.get('job_name')}  {j.get('status')}")
        return 0

    if not jobs:
        print("[down] nothing to tear down.")
        return 0

    for j in jobs:
        name = j.get("job_name")  # consistent .get access (S10)
        sky.jobs.cancel(name=name)
        print(f"[down] cancelled {name}")
    print("[down] fleet scaled to zero — no idle GPU spend.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
