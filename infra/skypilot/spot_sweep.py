#!/usr/bin/env python3
"""Cheapest-GPU spot sweep across Cairn's launchable AWS regions — RUN BEFORE ANY GPU TASK.

Spot prices move, a lot: over a few days in 2026-06 the cheapest g7e flipped Seoul → Spain → Ohio.
So we re-check every launch instead of trusting a remembered "cheapest region". This is read-only
(DescribeSpotPriceHistory, no spend) and only sweeps the regions we can actually launch in (the
IAM-allowed set). Output: current spot $/box per region for each GPU type, cheapest first.

    python3 infra/skypilot/spot_sweep.py                      # default GPU types, all launchable regions
    python3 infra/skypilot/spot_sweep.py g7e.2xlarge          # just the V4 Blackwell box
    python3 infra/skypilot/spot_sweep.py g7e.2xlarge --boxes 4   # + a fleet-total $/hr column
    python3 infra/skypilot/spot_sweep.py --regions us-east-2,eu-south-2 g6.xlarge

Price is NOT the whole story — before committing a region also confirm: G-spot QUOTA (>= vCPUs needed),
a HEALTHY default VPC (Ohio/Oregon IGWs were detached, now fixed), and the regional DLAMI id (AMIs are
per-region — see infra/skypilot/cairn-dsv4.sky.yaml). This is the price-discovery step, run first.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys

# Regions Cairn can launch in = the cairn-skypilot IAM allow-list (mutating actions).
LAUNCHABLE = ["us-east-2", "us-west-2", "eu-south-2", "ap-northeast-2"]
# Default GPU boxes we actually use: V4/headline Blackwell + the cheap L4 workhorse.
DEFAULT_TYPES = ["g7e.2xlarge", "g6.xlarge"]


def sweep(session, regions, types, lookback_h=6):
    """region × type → cheapest current Linux/UNIX spot price (latest per AZ, min across AZs)."""
    start = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=lookback_h)
    rows = {t: [] for t in types}
    for region in regions:
        ec2 = session.client("ec2", region_name=region)
        for t in types:
            try:
                # NB: query ONE instance type per call — a multi-type DescribeSpotPriceHistory
                # intermittently returns [] (observed 2026-06-24).
                hist = ec2.describe_spot_price_history(
                    InstanceTypes=[t], ProductDescriptions=["Linux/UNIX"],
                    StartTime=start, MaxResults=200,
                )["SpotPriceHistory"]
            except Exception as e:  # noqa: BLE001 — surface per-cell errors, keep sweeping
                print(f"  ! {region}/{t}: {e}", file=sys.stderr)
                continue
            latest_per_az = {}
            for h in hist:
                az = h["AvailabilityZone"]
                if az not in latest_per_az or h["Timestamp"] > latest_per_az[az]["Timestamp"]:
                    latest_per_az[az] = h
            if not latest_per_az:
                continue
            best = min(latest_per_az.values(), key=lambda h: float(h["SpotPrice"]))
            rows[t].append((float(best["SpotPrice"]), region, best["AvailabilityZone"], best["Timestamp"]))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("types", nargs="*", help=f"instance types (default: {' '.join(DEFAULT_TYPES)})")
    ap.add_argument("--regions", default=",".join(LAUNCHABLE), help="comma list (default: all launchable)")
    ap.add_argument("--profile", default="cairn-skypilot", help="AWS profile (default: %(default)s)")
    ap.add_argument("--boxes", type=int, default=1, help="fleet size → adds a total $/hr column")
    args = ap.parse_args()

    try:
        import boto3
    except ImportError:
        sys.exit("boto3 required — run with the cairn-sky venv python, or `pip install boto3`.")
    try:
        session = boto3.Session(profile_name=args.profile)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"could not open AWS profile '{args.profile}': {e}")

    types = args.types or DEFAULT_TYPES
    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    rows = sweep(session, regions, types)

    overall = []
    for t in types:
        data = sorted(rows.get(t, []), key=lambda r: r[0])
        print(f"\n=== {t}  (Linux/UNIX spot, cheapest first) ===")
        if not data:
            print("  (no spot offering found in the swept regions)")
            continue
        for i, (price, region, az, ts) in enumerate(data):
            tot = f"   fleet×{args.boxes} = ${price * args.boxes:.2f}/hr" if args.boxes > 1 else ""
            mark = "   <- cheapest" if i == 0 else ""
            print(f"  ${price:.4f}/box  {region:<15} {az:<16} @{ts:%m-%d %H:%MZ}{tot}{mark}")
        overall.append((data[0][0], t, data[0][1]))
    if len(overall) > 1:
        print("\n--- cheapest region per type ---")
        for price, t, region in overall:
            print(f"  {t:<14} ${price:.4f}/box  {region}")


if __name__ == "__main__":
    main()
