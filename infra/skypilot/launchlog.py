#!/usr/bin/env python3
"""Cairn GPU-box launch ledger — a LOCAL mirror of AWS instance lifetimes, for cost reconciliation.

AWS is the source of truth for ACTUAL cost (Cost Explorer, filtered by the `cairn` cost-allocation tag
— see cost-report.sh). This is the local ledger: every `sky launch` / `sky down` appends an event so we
can reconcile "how many boxes, which type, how long, ~how much" against the AWS bill (actual vs ours).

    python launchlog.py up   --cluster cairn-dev --instance g6.xlarge --region eu-south-2 --gpu L4 --price 0.16 --spot
    python launchlog.py down --cluster cairn-dev
    python launchlog.py report

Append-only log: infra/skypilot/launch-log.jsonl (one JSON event per line). Estimated cost = duration ×
--price (the spot quote at launch); the AWS report is the actual figure to reconcile against.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib

LOG = pathlib.Path(__file__).resolve().parent / "launch-log.jsonl"


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _parse(ts: str) -> dt.datetime:
    return dt.datetime.fromisoformat(ts)


def _append(rec: dict) -> None:
    rec = {"ts": _now(), **rec}
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def _events() -> list:
    if not LOG.exists():
        return []
    return [json.loads(line) for line in LOG.read_text().splitlines() if line.strip()]


def cmd_up(a: argparse.Namespace) -> None:
    _append({"event": "up", "cluster": a.cluster, "instance": a.instance, "region": a.region,
             "gpu": a.gpu, "price_hr": a.price, "spot": a.spot, "note": a.note})
    print(f"[launchlog] UP   {a.cluster}  ({a.instance}, {a.region}, ${a.price}/hr{' spot' if a.spot else ''})")


def cmd_down(a: argparse.Namespace) -> None:
    _append({"event": "down", "cluster": a.cluster, "note": a.note})
    print(f"[launchlog] DOWN {a.cluster}")


def _sessions(evs: list):
    """Pair each `up` with the next `down` for the same cluster (FIFO). Returns (paired, still_open)."""
    open_up: dict = {}
    paired = []
    for e in evs:
        c = e.get("cluster", "")
        if e["event"] == "up":
            open_up.setdefault(c, []).append(e)
        elif e["event"] == "down" and open_up.get(c):
            paired.append((open_up[c].pop(0), e))
    still_open = [u for ups in open_up.values() for u in ups]
    return paired, still_open


def cmd_report(a: argparse.Namespace) -> None:
    paired, still_open = _sessions(_events())
    print(f"{'cluster':14} {'instance':13} {'region':12} {'start (UTC)':17} {'hours':>7} {'~cost':>8}")
    print("-" * 75)
    total = 0.0
    rows = [(u, d, False) for (u, d) in paired] + [(u, None, True) for u in still_open]
    rows.sort(key=lambda r: r[0]["ts"])
    for up, down, running in rows:
        end = _parse(_now()) if running else _parse(down["ts"])
        hrs = (end - _parse(up["ts"])).total_seconds() / 3600
        cost = hrs * (up.get("price_hr") or 0.0)
        total += cost
        tag = " RUNNING" if running else ""
        print(f"{up['cluster']:14} {up.get('instance',''):13} {up.get('region',''):12} "
              f"{up['ts'][:16]:17} {hrs:6.2f}h ${cost:7.2f}{tag}")
    print("-" * 75)
    print(f"{'sessions: ' + str(len(rows)):>57}  TOTAL ~${total:7.2f}")
    print("\nACTUAL cost (reconcile against this): AWS Cost Explorer filtered by tag cairn=true —")
    print("  run infra/skypilot/cost-report.sh (needs billing perms + the activated cost-allocation tag).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    up = sub.add_parser("up", help="log a box launch")
    up.add_argument("--cluster", required=True)
    up.add_argument("--instance", required=True)
    up.add_argument("--region", required=True)
    up.add_argument("--gpu", default="")
    up.add_argument("--price", type=float, default=0.0, help="quoted $/hr at launch (for the estimate)")
    up.add_argument("--spot", action="store_true")
    up.add_argument("--note", default="")
    up.set_defaults(func=cmd_up)
    dn = sub.add_parser("down", help="log a box teardown")
    dn.add_argument("--cluster", required=True)
    dn.add_argument("--note", default="")
    dn.set_defaults(func=cmd_down)
    rep = sub.add_parser("report", help="render the ledger + estimated totals")
    rep.set_defaults(func=cmd_report)
    a = ap.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
