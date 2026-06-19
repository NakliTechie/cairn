"""End-to-end: start the demo HTTP server and round-trip real requests through the
whole rung-1 stack (gateway → scheduler → mock pipeline → recovery)."""

import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request

import pytest

_DEMO = pathlib.Path(__file__).resolve().parents[2] / "demo"
if str(_DEMO) not in sys.path:
    sys.path.insert(0, str(_DEMO))

from server import API_KEY, make_server  # noqa: E402


@pytest.fixture(scope="module")
def base_url():
    srv = make_server(0)  # ephemeral port
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        srv.shutdown()


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, json.loads(r.read())


def _post(url, body, auth=None):
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    if auth:
        req.add_header("authorization", auth)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_version_and_health(base_url):
    status, body = _get(base_url + "/version")
    assert status == 200 and body["service"] == "cairn"
    status, body = _get(base_url + "/v1/health")
    assert status == 200 and body["status"] == "ok"


def test_chat_completion_roundtrip(base_url):
    status, body = _post(
        base_url + "/v1/chat/completions",
        {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "hello cairn"}], "max_tokens": 10},
        auth=f"Bearer {API_KEY}",
    )
    assert status == 200
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["usage"]["completion_tokens"] == 10


def test_auth_required(base_url):
    status, body = _post(
        base_url + "/v1/chat/completions",
        {"model": "gpt-oss-120b", "messages": [{"role": "user", "content": "x"}]},
    )
    assert status == 401 and "error" in body


def test_unknown_model_404(base_url):
    status, _ = _post(
        base_url + "/v1/chat/completions",
        {"model": "no-such-model", "messages": [{"role": "user", "content": "x"}]},
        auth=f"Bearer {API_KEY}",
    )
    assert status == 404


def test_scenario_runs_with_recovery(base_url):
    status, body = _get(base_url + "/demo/scenario")
    assert status == 200
    assert body["completed"] == body["k_streams"]      # every stream finished
    assert body["avg_occupancy"] > 0.5                  # multi-stream filled the pipe
    assert len(body["recovery"]) == 1                   # the induced eviction happened
    assert body["recovery"][0]["policy"] == "reassign"
