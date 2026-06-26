#!/usr/bin/env python3
"""Watch for KILLED Cairn boxes (spot reclaim / stop) and alert.

Reconciles `sky status` against the launch ledger (launch-log.jsonl): any cluster the ledger has OPEN
(an `up` with no later `down`) that `sky status` no longer lists as UP was killed → auto-log a DOWN
(keeps the ledger + cost honest even when a box vanishes on its own), fire a desktop + terminal alert,
and refresh the live report. Spot reclaims are silent otherwise — this is the safety net.

    python infra/skypilot/watch.py check                  # one-shot reconcile + alert (cron-able)
    python infra/skypilot/watch.py loop --interval 150     # keep watching (default 150s)
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEDGER = HERE / "launch-log.jsonl"
SKY = os.path.expanduser("~/.cairn-sky-venv/bin/sky")


def _sky_up():
    """Set of cluster names currently UP per `sky status` (refreshed). None if the query failed —
    so a transient sky/network hiccup never false-alarms a 'killed' box."""
    try:
        out = subprocess.run([SKY, "status", "--refresh"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        print(f"watch: sky status unavailable ({e}) — skipping this tick", flush=True)
        return None
    up = set()
    for line in out.splitlines():
        parts = line.split()
        if "UP" in parts and parts and parts[0] not in ("NAME", "Clusters"):
            up.add(parts[0])
    return up


def _open_sessions():
    """Clusters the ledger believes are still up (sum of up=+1 / down=-1 per cluster > 0)."""
    if not LEDGER.exists():
        return []
    net: dict = {}
    for line in LEDGER.read_text().splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        c = e.get("cluster", "")
        net[c] = net.get(c, 0) + (1 if e.get("event") == "up" else -1)
    return [c for c, n in net.items() if n > 0]


def _alert(msg: str) -> None:
    print(f"⚠  {msg}", flush=True)
    try:  # macOS desktop notification (best-effort; harmless if unavailable)
        subprocess.run(["osascript", "-e",
                        f'display notification "{msg}" with title "Cairn watch" sound name "Basso"'],
                       timeout=10, capture_output=True)
    except Exception:
        pass
    try:  # signal the HUMAN remotely (ntfy/Slack/CallMeBot via CAIRN_ALERT_WEBHOOK / CALLMEBOT_*) — the
          # abrupt-death safety net: the box-side notice can't fire if AWS kills it without the ~2-min warning.
        sys.path.insert(0, str(ROOT))
        from cairn_node import notify
        notify.alert(msg, title="Cairn watch")
    except Exception:
        pass


def _run(*args) -> None:
    try:
        subprocess.run([sys.executable, *args], timeout=60, capture_output=True)
    except Exception:
        pass


def check() -> list:
    up = _sky_up()
    if up is None:
        return []
    openc = _open_sessions()
    killed = [c for c in openc if c not in up]
    for c in killed:
        _run(str(HERE / "launchlog.py"), "down", "--cluster", c, "--note", "detected killed (sky status)")
        _alert(f"box {c} was KILLED (spot reclaim / stop) — logged the teardown")
    if killed:
        _run(str(ROOT / "report" / "gen.py"), "build")  # refresh the cost tab
    print(f"watch: up={sorted(up)} ledger_open={openc} killed={killed}", flush=True)
    return killed


def loop(interval: int) -> None:
    print(f"watch: polling every {interval}s (Ctrl-C to stop)", flush=True)
    while True:
        check()
        time.sleep(interval)


def wait(cluster: str, interval: int = 60, max_wait: int = 7200) -> None:
    """BLOCK until `cluster` is no longer UP, then exit. Run in the BACKGROUND so its completion WAKES the
    agent the instant the box dies (the harness re-invokes on background-command exit) — the active-wake
    counterpart to `loop` (which only prints + desktop-alerts). Waits for the box to appear UP first, so it
    never false-fires during provisioning. Exit 3 = box died; 0 = max_wait elapsed (box still up)."""
    print(f"watch-wait: blocking until {cluster} dies (poll ~{interval}s, max {max_wait}s)", flush=True)
    waited = 0
    seen_up = False
    while waited < max_wait:
        up = _sky_up()
        if up is not None:
            if cluster in up:
                seen_up = True
            elif seen_up:
                check()                                   # reconcile: auto-log the down + desktop alert
                print(f"watch-wait: {cluster} is GONE — exiting to WAKE the agent", flush=True)
                raise SystemExit(3)
        time.sleep(interval)
        waited += interval
    print(f"watch-wait: max_wait {max_wait}s elapsed ({cluster} still up)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check", help="one-shot reconcile + alert").set_defaults(fn=lambda a: check())
    lp = sub.add_parser("loop", help="poll continuously")
    lp.add_argument("--interval", type=int, default=150)
    lp.set_defaults(fn=lambda a: loop(a.interval))
    wp = sub.add_parser("wait", help="block until a cluster dies, then exit (WAKES the agent)")
    wp.add_argument("--cluster", required=True)
    wp.add_argument("--interval", type=int, default=60)
    wp.add_argument("--max-wait", type=int, default=7200)
    wp.set_defaults(fn=lambda a: wait(a.cluster, a.interval, a.max_wait))
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
