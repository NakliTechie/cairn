"""cairn_node.recovery_fleet — the LIVE driver for the Path-1 recovery test, run on rank0 alongside the
entry node. Connects to the local entry, accepts the tail + the pre-warmed spare (separate sinks), then
decodes with recovery: if the tail dies mid-stream (CAIRN_DIE_AFTER on rank1) the driver re-stitches to
the spare, replays, resumes. Prints `TOKENS [...] MTTR ...`. Run it once with the tail NOT dying (the
reference) and once with it dying (the recovery) and compare the two token lists.

    python -m cairn_node.recovery_fleet --spare-host <ip2> --prompt 1,5,9,3,7,2 --n-new 8
"""
from __future__ import annotations

import argparse
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
    a = ap.parse_args()

    wire.key_from_env("SHARD_PSK")
    prompt = [int(x) for x in a.prompt.split(",") if x.strip()]

    sink_t = _listen(a.bind_host, a.tail_sink)
    sink_s = _listen(a.bind_host, a.spare_sink)
    head = LanEdge(a.entry_host, a.entry_port); _connect_retry(head)    # driver -> entry (local)
    ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)           # tail  -> driver
    cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)          # spare -> driver (standby)

    toks, mttr = decode_with_recovery(head, tail_e, spare_e, a.spare_host, a.spare_port,
                                      prompt, a.n_new, seq=a.seq)
    print("TOKENS", toks, "MTTR", ("%.3fs" % mttr) if mttr is not None else "none(no-death)")
    try:
        head.send({"op": "stop"})
    except Exception:
        pass


if __name__ == "__main__":
    main()
