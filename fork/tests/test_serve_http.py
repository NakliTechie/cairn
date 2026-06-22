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
