"""Proactive spot-drain — GRACEFUL migration to the warm spare BEFORE the node dies (no dropped tokens).

Reactive recovery (test_serve_http / recovery.py) waits for the node to DIE, then re-stitches + replays —
correct, but there's a brief gap. A real spot reclaim isn't a surprise: AWS posts a ~2-min interruption
notice on IMDS first. So a node can watch for its OWN notice and tell the driver "I'm draining" WHILE STILL
ALIVE; the driver pre-emptively re-stitches the entry to the warm spare + replays (priming the spare's KV)
and the in-flight request continues seamlessly — same set_next + replay mechanism, the difference is the
TRIGGER (a draining signal, not a death) and that there's no gap. The drained node DOES then exit — the
re-stitch closes its edge_in — so "graceful" is about the STREAM migrating, not the node surviving (the
node was alive only long enough to emit the signal); asserted below.

This proves it on the CPU mock — no GPU, no spend:
  • entry -> tail -> driver, plus a pre-warmed spare, run a clean NO-DRAIN reference decode (the oracle).
  • Re-stand the fleet; spawn the tail with CAIRN_SPOT_TEST_FILE=<sentinel> (a local stand-in for IMDS —
    the tail's watch thread flips `draining` the instant the file appears). Decode again; mid-stream (on
    the driver's first committed token) the test `touch`es the sentinel. The tail emits {"op":"draining"}
    in place of its next forward result; the driver migrates to the spare and finishes the stream.
  • Assert: output is BIT-IDENTICAL to the no-drain reference, NO exception was raised, and on_event shows
    the PROACTIVE path fired (a "draining" event + a "recovered" event) and NOT the reactive "death" path —
    i.e. the migration was graceful (no EDGE_ERRORS death was ever needed) — and THEN the drained tail
    exits (the re-stitch closes its edge_in): graceful is about the stream, the tail does not survive.

torch/cryptography-gated (the wire pulls in torch)."""
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "fork")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("cryptography")
os.environ.setdefault("SHARD_PSK", "cairn-test-psk")

NL, CUT = 8, 4
PROMPT, NNEW = [1, 5, 9, 3, 7, 2], 8


def _spawn(ls, le, lp, np_, *, spot_file=None, capture=False):
    """Spawn one mock cairn_node.serve: layers [ls,le), listen :lp, dial next 127.0.0.1:np_. If `spot_file`
    is set, the node arms a drain watcher on that sentinel (the test's IMDS stand-in) — touching the file
    makes the node start draining, with NO real EC2/IMDS. `capture=True` pipes the node's stderr so a test
    can inspect its exit trace (used to assert the drained tail exits cleanly, with NO traceback)."""
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    env.pop("CAIRN_DIE_AFTER", None)                       # nobody dies in the proactive path
    if spot_file:
        env["CAIRN_SPOT_TEST_FILE"] = spot_file
        env["CAIRN_SPOT_WATCH_INTERVAL"] = "0.001"         # tight poll so the drain lands mid-(fast mock)-decode
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", "mock", "--model", "mock:8",
         "--layer-start", str(ls), "--layer-end", str(le), "--device", "cpu", "--listen-port", str(lp),
         "--bind-host", "127.0.0.1", "--next-host", "127.0.0.1", "--next-port", str(np_)],
        cwd=str(_ROOT), env=env,
        stderr=subprocess.PIPE if capture else None)


def _sink(port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(("127.0.0.1", port))
    s.listen(1)
    return s


def _reference():
    """No-drain oracle: entry+tail (no spare), clean decode. This is what the proactive run must match."""
    from cairn_node.pipeline import run_pipeline
    return run_pipeline("mock", "mock:8", NL, [CUT], PROMPT, NNEW, device="cpu", base_port=7960)


def test_proactive_drain_migrates_to_spare_gracefully(tmp_path):
    from shard.transport import LanEdge
    from shard import wire
    wire.key_from_env("SHARD_PSK")
    from cairn_node.serve import _connect_retry
    from cairn_node.recovery import decode_with_recovery

    ref = _reference()                                          # the bit-identical target
    assert len(ref) == NNEW

    sentinel = str(tmp_path / "spot-notice")                    # IMDS stand-in; absent until the test touches it
    b = 7970
    pE, pT, pSpare, sinkT, sinkS = b, b + 1, b + 2, b + 3, b + 4
    # entry 0..CUT (-> tail); tail CUT..NL (-> driver tail-sink), arms the drain watcher on `sentinel`;
    # spare CUT..NL (-> driver spare-sink), pre-warmed standby. NOBODY has CAIRN_DIE_AFTER — no death.
    entry = _spawn(0, CUT, pE, pT)
    tail = _spawn(CUT, NL, pT, sinkT, spot_file=sentinel)
    spare = _spawn(CUT, NL, pSpare, sinkS)
    procs = [entry, tail, spare]
    sink_t = _sink(sinkT)
    sink_s = _sink(sinkS)
    head = None
    try:
        head = LanEdge("127.0.0.1", pE); _connect_retry(head)          # driver -> entry
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)      # tail  -> driver
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)     # spare -> driver (standby)

        events = []

        def _on(kind, *p):
            events.append((kind, p))
            # Trigger the drain MID-DECODE: the instant the driver commits its first token, post the spot
            # notice (touch the sentinel). The mock decode is microseconds/token, so to land the drain mid-
            # stream DETERMINISTICALLY we (a) run the tail's watcher at a 1ms poll and (b) block HERE — inside
            # the synchronous decode loop — until the tail's edge falls quiet, i.e. the tail has consumed the
            # in-flight forward and is now sitting on the drain path (its result will be {"op":"draining"}).
            # A short bounded wait > the poll interval guarantees the flag is flipped before the next forward.
            if kind == "tok" and p[0] == 1 and not os.path.exists(sentinel):
                pathlib.Path(sentinel).write_text("going down")
                time.sleep(0.05)                          # >> the 1ms poll: draining is set before the next send

        out, mttr = decode_with_recovery(head, tail_e, spare_e, "127.0.0.1", pSpare,
                                         PROMPT, NNEW, seq="drain0", on_event=_on)

        # 1) GRACEFUL + correct: same tokens as the no-drain reference, and we did migrate (mttr is set).
        assert out == ref, f"proactive-drain output {out} != no-drain reference {ref}"
        assert mttr is not None, "expected a pre-emptive migration onto the spare (mttr should be set)"

        # 2) It was the PROACTIVE path, not the reactive one: a draining event fired, a recovered event
        #    fired, and NO death event ever did (no EDGE_ERRORS — the old node never had to die).
        kinds = [k for k, _ in events]
        assert "draining" in kinds, f"no proactive 'draining' event — drain path did not fire ({kinds})"
        assert "recovered" in kinds, f"no 'recovered' event after the drain ({kinds})"
        assert "death" not in kinds, f"a reactive 'death' fired — migration was NOT graceful ({kinds})"

        # 3) "Graceful" is about the STREAM, not the tail: the drained tail EXITS shortly after the migration.
        #    The driver's set_next (in _recover_to_spare) closes the entry->tail edge, so the tail's next
        #    edge_in.recv() hits peer-closed and the process exits (serve.py). It does NOT outlive the
        #    migration — it was alive only long enough to emit the draining signal. (The earlier
        #    `assert tail.poll() is None` passed only because it sampled the one instant before the FIN had
        #    propagated; the tail exits ~150ms later — measured CPU repro, 2026-06-23.)
        for _ in range(300):                                    # up to ~3s for the post-re-stitch peer-closed
            if tail.poll() is not None:
                break
            time.sleep(0.01)
        assert tail.poll() is not None, "drained tail did not exit after the re-stitch closed its edge_in"

        # 4) The committed-token count at the drain is consistent with where we triggered it.
        n_at_drain = next(p[0] for k, p in events if k == "draining")
        assert 0 < n_at_drain < NNEW

        head.send({"op": "stop"})
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


def test_proactive_drain_tail_exits_cleanly_rc0(tmp_path):
    """Fix (a): the drained tail must exit CLEANLY (rc=0), not on an unhandled ConnectionError (rc=1).

    The sibling test above asserts the MIGRATION is graceful + bit-identical and that the tail EXITS
    (poll() is not None) — but it is rc-AGNOSTIC, so it stays green whether the exit is clean or a
    swallowed traceback. This test pins the exit DOWN: after the driver re-stitches the entry to the
    spare (closing the tail's edge_in), serve.py's wire loop sees the EXPECTED peer-closed while
    `drained_sent` is True, logs a clean line on STDOUT, and breaks → process rc=0 with NO
    ConnectionError traceback on STDERR. (Before the fix this was an unhandled ConnectionError → rc=1,
    only masked by the serve yaml's `exit 0`.) We capture the tail's stderr to prove the traceback is
    gone."""
    from shard.transport import LanEdge
    from shard import wire
    wire.key_from_env("SHARD_PSK")
    from cairn_node.serve import _connect_retry
    from cairn_node.recovery import decode_with_recovery

    sentinel = str(tmp_path / "spot-notice")                    # IMDS stand-in; absent until we touch it
    b = 7980
    pE, pT, pSpare, sinkT, sinkS = b, b + 1, b + 2, b + 3, b + 4
    entry = _spawn(0, CUT, pE, pT)
    tail = _spawn(CUT, NL, pT, sinkT, spot_file=sentinel, capture=True)   # capture stderr to inspect the exit
    spare = _spawn(CUT, NL, pSpare, sinkS)
    procs = [entry, tail, spare]
    sink_t = _sink(sinkT)
    sink_s = _sink(sinkS)
    head = None
    try:
        head = LanEdge("127.0.0.1", pE); _connect_retry(head)          # driver -> entry
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)      # tail  -> driver
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)     # spare -> driver (standby)

        def _on(kind, *p):
            # Trigger the drain mid-decode (same pattern as the sibling test): on the first committed
            # token, post the spot notice and block briefly so `draining` is set before the next forward.
            if kind == "tok" and p[0] == 1 and not os.path.exists(sentinel):
                pathlib.Path(sentinel).write_text("going down")
                time.sleep(0.05)                                       # >> the 1ms poll

        out, mttr = decode_with_recovery(head, tail_e, spare_e, "127.0.0.1", pSpare,
                                         PROMPT, NNEW, seq="drainrc", on_event=_on)
        assert mttr is not None, "drain did not migrate onto the spare — test did not exercise the path"
        assert len(out) == NNEW

        # The drained tail exits shortly after the re-stitch closes its edge_in — and now it exits rc=0.
        for _ in range(300):                                           # up to ~3s for the post-re-stitch peer-closed
            if tail.poll() is not None:
                break
            time.sleep(0.01)
        assert tail.poll() == 0, f"drained tail should exit cleanly (rc=0); got rc={tail.poll()}"

        # The clean-exit path leaves NO traceback on stderr (the old unhandled ConnectionError did). The
        # tail has exited, so its stderr pipe is at EOF and read() returns immediately.
        err = tail.stderr.read().decode(errors="replace") if tail.stderr else ""
        assert "Traceback" not in err and "ConnectionError" not in err, \
            f"drained tail exited with an error trace on stderr (fix (a) regressed):\n{err}"

        head.send({"op": "stop"})
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
