"""cairn_node.recovery_fleet — the LIVE driver for the Path-1 recovery test, run on rank0 alongside the
entry node. Connects to the local entry, accepts the tail + the pre-warmed spare (separate sinks), then
decodes with recovery: if the tail dies mid-stream (CAIRN_DIE_AFTER on rank1) the driver re-stitches to
the spare, replays, resumes. Prints `TOKENS [...] MTTR ...`. Run it once with the tail NOT dying (the
reference) and once with it dying (the recovery) and compare the two token lists.

    python -m cairn_node.recovery_fleet --spare-host <ip2> --prompt 1,5,9,3,7,2 --n-new 8
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "fork") not in sys.path:
    sys.path.insert(0, str(_ROOT / "fork"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shard.transport import LanEdge  # noqa: E402
from shard import wire  # noqa: E402
from cairn_node.serve import _connect_retry, _listen  # noqa: E402
from cairn_node.recovery import decode_with_recovery  # noqa: E402


class _FlushLog:
    """Durable progress log: every line is flushed + fsync'd to an on-box file AND echoed to stdout.
    Buffered stdout over `sky exec` is LOST when the box/ssh dies mid-run — that is how the FIRST live
    recovery run lost its verdict (it produced nothing before teardown). This file survives on the box
    for a later `sky exec '... cat <log>'`, so the run is never silent (truthful-run discipline)."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._fh = open(path, "a", buffering=1)        # line-buffered

    def __call__(self, msg: str) -> None:
        print(msg, flush=True)
        self._fh.write(msg + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())                    # force it to disk — survives a hard kill

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry-host", default="127.0.0.1")
    ap.add_argument("--entry-port", type=int, default=7777)
    ap.add_argument("--tail-sink", type=int, default=7779)     # the tail dials back here
    ap.add_argument("--spare-sink", type=int, default=7780)    # the spare dials back here
    ap.add_argument("--bind-host", default="0.0.0.0")
    ap.add_argument("--spare-host", required=True)             # rank2 IP — the entry re-points here on a death
    ap.add_argument("--spare-port", type=int, default=7777)
    ap.add_argument("--prompt", default="1,5,9,3,7,2")
    ap.add_argument("--n-new", type=int, default=8)
    ap.add_argument("--seq", default="rec")
    ap.add_argument("--log", default="/tmp/cairn-recovery.log")        # flushed on-box verdict (survives a kill)
    a = ap.parse_args()

    wire.key_from_env("SHARD_PSK")
    prompt = [int(x) for x in a.prompt.split(",") if x.strip()]
    log = _FlushLog(a.log)
    log(f"[recovery] start seq={a.seq} prompt={prompt} n_new={a.n_new} "
        f"spare={a.spare_host}:{a.spare_port} die_after(env)={os.environ.get('CAIRN_DIE_AFTER', '?')}")

    sink_t = _listen(a.bind_host, a.tail_sink)
    sink_s = _listen(a.bind_host, a.spare_sink)
    head = LanEdge(a.entry_host, a.entry_port); _connect_retry(head)    # driver -> entry (local)
    log(f"[recovery] sinks up (tail :{a.tail_sink}, spare :{a.spare_sink}); entry dialed — waiting for tail")
    ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)           # tail  -> driver
    log("[recovery] tail connected back — waiting for spare (standby)")
    cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)          # spare -> driver (standby)
    log("[recovery] spare connected — decoding with recovery armed")

    def _on_event(kind, *p):
        if kind == "tok":
            log(f"[recovery] tok {p[0]}/{a.n_new} = {p[1]}")
        elif kind == "death":
            log(f"[recovery] *** NODE DEATH after {p[0]} committed tokens — re-stitch entry -> spare "
                f"{a.spare_host}:{a.spare_port}, replay {len(prompt)}+{p[0]} ids")
        elif kind == "recovered":
            log(f"[recovery] *** RECOVERED in {p[0]:.3f}s — resumed at tok {p[2]} = {p[1]}")

    toks, mttr = decode_with_recovery(head, tail_e, spare_e, a.spare_host, a.spare_port,
                                      prompt, a.n_new, seq=a.seq, on_event=_on_event)
    log("TOKENS %s MTTR %s" % (toks, ("%.3fs" % mttr) if mttr is not None else "none(no-death)"))
    try:
        head.send({"op": "stop"})
    except Exception:
        pass
    log("[recovery] driver done")
    log.close()


if __name__ == "__main__":
    main()
