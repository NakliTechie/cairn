"""Cairn BYOK endpoint (cairn_node.serve_http) — the OpenAI-compatible front that drives the fleet.

On the CPU mock this proves the FULL request plumbing end to end: real HTTP → Bearer (BYOK) auth → parse
→ tokenize → drive the wire pipeline → detokenize → OpenAI chat.completion (+ a 401 on a bad key). Real
HF tokenizer + real sglang inference is the GPU path; recovery (warm spare) reuses the proven
decode_with_recovery. torch/cryptography-gated."""
import json
import os
import pathlib
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "fork"), str(_ROOT / "scheduler" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("cryptography")
os.environ.setdefault("SHARD_PSK", "cairn-test-psk")


def _spawn(ls, le, lp, np_):
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", "mock", "--model", "mock:8",
         "--layer-start", str(ls), "--layer-end", str(le), "--device", "cpu", "--listen-port", str(lp),
         "--bind-host", "127.0.0.1", "--next-host", "127.0.0.1", "--next-port", str(np_)],
        cwd=str(_ROOT), env=env)


def _post(url, body, key=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    if key:
        req.add_header("authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_serve_http_mock_roundtrip():
    from cairn_node.serve_http import FleetEngine, connect_fleet, make_server
    from shard import wire
    wire.key_from_env("SHARD_PSK")

    base = 7840
    entry = _spawn(0, 4, base, base + 1)            # entry 0..4 -> tail :base+1
    tail = _spawn(4, 8, base + 1, base + 2)          # tail 4..8 -> driver sink :base+2
    procs = [entry, tail]
    head = srv = None
    try:
        head, tail_e, _spares, _sink = connect_fleet("127.0.0.1", base, base + 2, bind_host="127.0.0.1")
        eng = FleetEngine("mock:8", head, tail_e, api_key="sk-test")
        srv = make_server(eng, "127.0.0.1", base + 5)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        time.sleep(0.3)
        url = f"http://127.0.0.1:{base + 5}/v1/chat/completions"
        body = {"model": "mock:8", "messages": [{"role": "user", "content": "hello"}], "max_tokens": 8}

        st, _ = _post(url, body)                                    # no key -> 401
        assert st == 401

        st, resp = _post(url, body, key="sk-test")                 # valid key -> 200 OpenAI shape
        assert st == 200, resp
        assert resp["object"] == "chat.completion"
        assert resp["choices"][0]["message"]["role"] == "assistant"
        assert len(resp["choices"][0]["message"]["content"]) > 0    # real (decoded) content, not empty
        assert resp["usage"]["completion_tokens"] == 8
    finally:
        if srv is not None:
            srv.shutdown()
        if head is not None:
            try:
                head.send({"op": "stop"})
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


def test_serve_http_recovers_on_abrupt_tail_kill():
    """The realistic spot-reclaim case (found live, 2026-06-22): ABRUPTLY SIGKILL the tail mid-serving —
    the endpoint must detect the death, re-stitch to the warm spare, and STILL answer the next request.
    The graceful CAIRN_DIE_AFTER path misses this: there the tail recv's the entry's send before exiting,
    so the entry never hits a broken-pipe send. An abrupt kill DOES break that send — and without the
    entry surviving it (serve.py wraps edge_out.send), it crashes and the driver can never re-stitch it,
    so recovery hangs forever (exactly the live symptom)."""
    from cairn_node.serve_http import FleetEngine, connect_fleet
    from cairn_scheduler.gateway import ChatRequest
    from shard import wire
    wire.key_from_env("SHARD_PSK")

    base = 7880
    entry = _spawn(0, 4, base, base + 1)             # entry 0..4 -> tail :base+1
    tail = _spawn(4, 8, base + 1, base + 3)           # tail 4..8 -> endpoint tail-sink :base+3
    spare = _spawn(4, 8, base + 2, base + 4)          # spare 4..8 -> endpoint spare-sink :base+4
    procs = [entry, tail, spare]
    head = None
    try:
        head, tail_e, _spares, _sink = connect_fleet("127.0.0.1", base, base + 3, bind_host="127.0.0.1",
                                                     spare_sink=base + 4, n_spares=1)
        assert _spares
        spare_e = _spares[0][0]
        eng = FleetEngine("mock:8", head, tail_e, spare=spare_e, spare_host="127.0.0.1", spare_port=base + 2)
        req = ChatRequest(model="mock:8", messages=[{"role": "user", "content": "hello"}], max_tokens=6)

        r1 = eng.complete(req)                                       # baseline — works, no recovery
        assert len(r1["choices"][0]["message"]["content"]) > 0 and "cairn" not in r1

        tail.kill()                                                 # ABRUPT death (SIGKILL ~ a spot reclaim)
        time.sleep(0.5)

        result = {}
        th = threading.Thread(target=lambda: result.setdefault("r", eng.complete(req)), daemon=True)
        th.start(); th.join(timeout=30)
        assert "r" in result, "endpoint HUNG after an abrupt tail kill — recovery did not complete"
        r2 = result["r"]
        assert len(r2["choices"][0]["message"]["content"]) > 0
        assert r2.get("cairn", {}).get("recovered") is True         # recovered onto the warm spare
    finally:
        if head is not None:
            try:
                head.send({"op": "stop"})
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


class _HalfOpenRelay:
    """A TCP relay :listen_port -> (dst_host, dst_port) that pumps both directions until frozen.

    On freeze() it simulates a RECLAIMED BOX: forwarding stops but BOTH sockets are left OPEN and IDLE —
    so after the tail process is killed, NO FIN/RST ever reaches the driver and the driver's tail-sink
    socket stays ESTABLISHED-but-silent (the half-open case). A driver recv() on it then blocks FOREVER
    unless a recv timeout is armed. This is exactly what localhost cannot produce on its own (a local
    SIGKILL FINs the peer instantly), and is why the plain-`tail.kill()` mock test passes while the live
    box hung. The relay process (the test) stays alive holding the driver socket open, so the only thing
    the driver can observe is silence — death is detectable solely via the supervised recv deadline."""

    def __init__(self, listen_port, dst_host, dst_port):
        self.dst = (dst_host, dst_port)
        self._frozen = threading.Event()
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind(("127.0.0.1", listen_port))
        self._srv.listen(1)
        self._inbound = None
        self._outbound = None
        threading.Thread(target=self._accept, daemon=True).start()

    def freeze(self):
        # Stop forwarding but DO NOT touch either socket — both stay open + idle → the driver-facing
        # socket is half-open (no FIN). Proven to block a no-timeout recv() forever.
        self._frozen.set()

    def _accept(self):
        inbound, _ = self._srv.accept()                 # the tail node dials the relay
        outbound = socket.create_connection(self.dst)   # the relay dials the driver's tail-sink
        self._inbound, self._outbound = inbound, outbound
        threading.Thread(target=self._pump, args=(inbound, outbound), daemon=True).start()
        threading.Thread(target=self._pump, args=(outbound, inbound), daemon=True).start()

    def _pump(self, a, b):
        try:
            while not self._frozen.is_set():
                a.settimeout(0.1)
                try:
                    data = a.recv(65536)
                except socket.timeout:
                    continue
                if not data:
                    break
                try:
                    b.sendall(data)
                except OSError:
                    break
        except OSError:
            pass
        # On freeze we never close either socket → the driver side stays half-open: no FIN to the driver.

    def close(self):
        self._frozen.set()
        for s in (self._inbound, self._outbound, self._srv):
            try:
                s.close()
            except (OSError, AttributeError):
                pass


def test_serve_http_recovers_on_HALF_OPEN_tail_death():
    """The REAL live hang (2026-06-22): between requests the tail box is reclaimed and the driver's
    tail-sink goes HALF-OPEN — TCP still ESTABLISHED, no FIN/RST ever arrives. The plain mock test
    (test_serve_http_recovers_on_abrupt_tail_kill) MISSES this because a local SIGKILL FINs the peer
    instantly, so the driver's next recv() gets ECONNRESET and recovery fires. Here a relay holds the
    driver socket open + silent after the tail dies, so the ONLY way the driver can notice the death is
    the supervised recv timeout. Without it (legacy: no timeout on from_socket edges) active.recv()
    blocks FOREVER and this thread never returns → the assert below fails (the live symptom). With the
    fix (CAIRN_EDGE_RECV_TIMEOUT on the driver's tail/spare sinks) the stall is caught as a death,
    recovery re-stitches to the warm spare, and the next request is answered."""
    from cairn_node.serve_http import FleetEngine
    from cairn_node.serve import _connect_retry, _listen
    from cairn_scheduler.gateway import ChatRequest
    from shard.transport import LanEdge
    from shard import wire
    wire.key_from_env("SHARD_PSK")

    base = 7900
    relay_port = base + 9
    entry = _spawn(0, 4, base, base + 1)              # entry 0..4 -> tail :base+1
    tail = _spawn(4, 8, base + 1, relay_port)          # tail 4..8 -> RELAY :base+9 -> driver tail-sink :base+3
    spare = _spawn(4, 8, base + 2, base + 4)           # spare 4..8 -> driver spare-sink :base+4
    procs = [entry, tail, spare]
    head = None
    sink_t = _listen("127.0.0.1", base + 3)
    sink_s = _listen("127.0.0.1", base + 4)
    relay = _HalfOpenRelay(relay_port, "127.0.0.1", base + 3)
    logs = []
    try:
        # SHORT recv deadline so the test is fast; this is the ONLY death-detection path here.
        head = LanEdge("127.0.0.1", base, recv_timeout=3.0); _connect_retry(head)
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct, recv_timeout=3.0)   # driver <- RELAY (will go half-open)
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs, recv_timeout=3.0)  # driver <- spare
        eng = FleetEngine("mock:8", head, tail_e, spare=spare_e, spare_host="127.0.0.1", spare_port=base + 2,
                          log=logs.append)              # capture the recovery trace to assert observability
        req = ChatRequest(model="mock:8", messages=[{"role": "user", "content": "hello"}], max_tokens=6)

        r1 = eng.complete(req)                                       # baseline — works, no recovery
        assert len(r1["choices"][0]["message"]["content"]) > 0 and "cairn" not in r1
        time.sleep(0.5)                                             # let the baseline's last frame fully drain the relay

        relay.freeze()                                             # tail box reclaimed: driver sink now HALF-OPEN (no FIN)
        tail.kill()
        time.sleep(0.5)                                            # let the kill settle (relay holds the driver socket open)

        result = {}
        th = threading.Thread(target=lambda: result.setdefault("r", eng.complete(req)), daemon=True)
        th.start(); th.join(timeout=30)
        assert "r" in result, ("endpoint HUNG on a HALF-OPEN tail death — recv() blocked forever, "
                               "death never detected, recovery never fired (the live symptom)")
        r2 = result["r"]
        assert len(r2["choices"][0]["message"]["content"]) > 0
        assert r2.get("cairn", {}).get("recovered") is True         # recovered onto the warm spare
        # observability: the recovery steps were written to the (flushed) serve log
        joined = "\n".join(logs)
        assert "TAIL DEATH detected" in joined and "RECOVERED" in joined, joined
    finally:
        relay.close()
        if head is not None:
            try:
                head.send({"op": "stop"})
            except Exception:
                pass
        for s in (sink_t, sink_s):
            try:
                s.close()
            except OSError:
                pass
        for p in procs:
            if p.poll() is None:
                p.kill()
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass


def test_serve_http_on_event_logs_proactive_drain():
    """Fix (b): a PROACTIVE drain must be VISIBLE in the serve log. Before this, _on_event only logged
    the reactive 'death' + 'recovered' events, so a graceful drain-before-death produced NO drain line —
    indistinguishable from a reactive death except by the ABSENCE of 'TAIL DEATH detected' (exactly why
    the 2026-06-23 live drain could not be classified as proactive vs reactive). Unit-level: no fleet
    needed (head/tail unused by _on_event), just the event hook + a capturing log sink."""
    from cairn_node.serve_http import FleetEngine
    from shard import wire
    wire.key_from_env("SHARD_PSK")

    logs = []
    eng = FleetEngine("mock:8", None, None, spare_host="10.0.0.9", spare_port=7777, log=logs.append)
    eng._on_event("draining", 3)                                # the proactive-drain event (3 committed tokens)
    joined = "\n".join(logs)
    assert "PROACTIVE DRAIN signalled after 3 committed tokens" in joined, joined
    assert "10.0.0.9:7777" in joined, joined                    # names the warm spare we re-stitch to
    # and it must NOT be misreported as the reactive death path
    assert "TAIL DEATH detected" not in joined, joined
