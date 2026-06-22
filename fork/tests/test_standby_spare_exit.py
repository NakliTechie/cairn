"""No-spend regression for the standby-spare hang.

A pre-warmed SPARE that is never promoted (a no-death run) must SELF-EXIT when the driver tears the
run down — NOT block on accept() forever. The forever-block is what hung the first live recovery run
(and its `sky launch`): the spare sat at accept() waiting for a prev that, with no death, never came.

The fix (serve.py): the node waits on its listen socket AND its downstream edge at once (select). If
the downstream (the driver) closes first, the run is over and the node exits cleanly. This reproduces
that on the CPU mock — no GPU, no spend — by standing up entry -> tail -> driver plus an idle spare,
running a clean no-death decode, then closing the driver's spare-sink and asserting the spare process
terminates. Pre-fix this test hangs at `spare.wait()` and times out.

torch/cryptography-gated (the wire pulls in torch)."""
import os
import pathlib
import socket
import subprocess
import sys
import time

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "fork")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("cryptography")
os.environ.setdefault("SHARD_PSK", "cairn-test-psk")


def _spawn(ls, le, lp, np_):
    """Spawn one mock cairn_node.serve: layers [ls,le), listen :lp, dial next 127.0.0.1:np_."""
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", "mock", "--model", "mock:8",
         "--layer-start", str(ls), "--layer-end", str(le), "--device", "cpu",
         "--listen-port", str(lp), "--bind-host", "127.0.0.1", "--next-host", "127.0.0.1",
         "--next-port", str(np_)],
        cwd=str(_ROOT), env=env)


def _sink(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s


def test_idle_standby_spare_exits_on_teardown():
    from shard.transport import LanEdge
    from shard import wire
    wire.key_from_env("SHARD_PSK")
    from cairn_node.pipeline import _drive

    NL, CUT, PROMPT, NNEW = 8, 4, [1, 5, 9, 3, 7, 2], 6
    # entry 0..CUT (:7821 -> tail :7822); tail CUT..NL (:7822 -> driver tail-sink :7824), NO death;
    # spare CUT..NL (:7823 -> driver spare-sink :7825): pre-warmed, NEVER promoted (there is no death).
    entry = _spawn(0, CUT, 7821, 7822)
    tail = _spawn(CUT, NL, 7822, 7824)
    spare = _spawn(CUT, NL, 7823, 7825)
    procs = [entry, tail, spare]
    sink_t = _sink(7824)
    sink_s = _sink(7825)
    try:
        head = LanEdge("127.0.0.1", 7821)
        for _ in range(600):                     # entry listens before its (instant, mock) load
            try:
                head.connect()
                break
            except OSError:
                time.sleep(0.05)
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)   # the spare's edge_out lands here

        out = _drive(head, tail_e, PROMPT, NNEW)                     # clean no-death decode (_drive sends stop)
        assert len(out) == NNEW

        # The spare was never promoted (no prev ever dialed its :7823). Tearing the driver down = closing
        # its spare-sink. That sends FIN to the spare's edge_out -> readable (EOF) -> its select() fires ->
        # it exits. Pre-fix it is parked in accept() and this wait() times out.
        spare_e.close()
        rc = spare.wait(timeout=10)
        assert rc == 0, f"standby spare did not exit cleanly on teardown (rc={rc})"

        head.close(); tail_e.close()
    finally:
        for s in (sink_t, sink_s):
            try:
                s.close()
            except Exception:
                pass
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass
