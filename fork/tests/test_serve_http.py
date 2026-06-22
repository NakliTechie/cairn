"""Cairn BYOK endpoint (cairn_node.serve_http) — the OpenAI-compatible front that drives the fleet.

On the CPU mock this proves the FULL request plumbing end to end: real HTTP → Bearer (BYOK) auth → parse
→ tokenize → drive the wire pipeline → detokenize → OpenAI chat.completion (+ a 401 on a bad key). Real
HF tokenizer + real sglang inference is the GPU path; recovery (warm spare) reuses the proven
decode_with_recovery. torch/cryptography-gated."""
import json
import os
import pathlib
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
        head, tail_e, _spare = connect_fleet("127.0.0.1", base, base + 2, bind_host="127.0.0.1")
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
        head, tail_e, spare_e = connect_fleet("127.0.0.1", base, base + 3, bind_host="127.0.0.1",
                                              spare_sink=base + 4)
        assert spare_e is not None
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
