"""cairn_node.spot_watch — listen for AWS's spot-interruption warning so recovery can be PROACTIVE.

A real spot reclaim isn't a surprise: AWS publishes a warning on the instance metadata service (IMDS)
~2 minutes before it pulls the box (and a rebalance recommendation even earlier). So instead of only
reacting AFTER a node dies (re-stitch + replay-rebuild, with a brief gap — `cairn_node.recovery`), a node
can watch for its own warning and tell the driver/control-plane to drain it onto the warm spare WHILE IT'S
STILL ALIVE — a graceful migration with no dropped tokens. The reactive path stays as the fallback (a
missed notice, or a death faster than the warning).

IMDSv2 (token-gated):
  - rebalance recommendation (earliest):  GET /latest/meta-data/events/recommendations/rebalance
  - interruption notice (~2 min warning):  GET /latest/meta-data/spot/instance-action -> {action, time}
Either present (HTTP 200) => this box is going down soon.

    python -m cairn_node.spot_watch            # poll this box's IMDS; print the notice when it appears
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

IMDS = "http://169.254.169.254"
TOKEN_TTL = 60
INTERRUPTION_PATH = "/latest/meta-data/spot/instance-action"
REBALANCE_PATH = "/latest/meta-data/events/recommendations/rebalance"


def _http(method: str, path: str, headers=None, timeout: float = 1.0):
    """One IMDS request → (status, body). 4xx/5xx (e.g. the 404 when no notice is pending) come back as a
    status, not an exception. A connection failure (not on EC2) raises — callers treat that as 'no notice'."""
    req = urllib.request.Request(IMDS + path, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, ""


def _token(http=_http):
    try:
        st, body = http("PUT", "/latest/api/token", {"X-aws-ec2-metadata-token-ttl-seconds": str(TOKEN_TTL)})
        return body if st == 200 and body else None
    except Exception:
        return None


def check_interruption(http=_http):
    """Return the interruption/rebalance notice dict (with `_kind`) if AWS has flagged THIS box, else None.
    `http(method, path, headers, timeout) -> (status, body)` is injectable so the logic is testable off-EC2."""
    tok = _token(http)
    hdr = {"X-aws-ec2-metadata-token": tok} if tok else {}
    for path, kind in ((INTERRUPTION_PATH, "interruption"), (REBALANCE_PATH, "rebalance")):
        try:
            st, body = http("GET", path, hdr)
        except Exception:
            continue                                         # not on EC2 / IMDS unreachable → treat as no notice
        if st == 200 and body.strip():
            try:
                d = json.loads(body)
            except Exception:
                d = {"raw": body}
            d["_kind"] = kind
            return d
    return None


def watch(on_signal, interval: float = 5.0, http=_http, stop=lambda: False):
    """Poll IMDS every `interval`s; the instant a notice appears, call `on_signal(notice)` ONCE and return it.
    `stop()` lets the caller end the loop. The doomed node wires on_signal to 'tell the driver: drain me'."""
    while not stop():
        notice = check_interruption(http)
        if notice is not None:
            on_signal(notice)
            return notice
        time.sleep(interval)
    return None


def main() -> None:
    print("[spot_watch] polling IMDS for a spot-interruption warning (Ctrl-C to stop)...", flush=True)
    n = watch(lambda notice: print(f"[spot_watch] *** GOING DOWN: {notice}", flush=True))
    print(f"[spot_watch] notice: {n}", flush=True)


if __name__ == "__main__":
    main()
