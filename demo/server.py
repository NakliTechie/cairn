"""End-to-end Cairn demo — the whole rung-1 stack, live and curl-able.

Wires the OpenAI-compatible gateway → the multi-stream scheduler → the no-GPU mock
pipeline → recovery, behind a stdlib HTTP server (no deps). It is the *control-plane +
data-plane logic* running together; only the SGLang block forward is mocked (rung 1).

    uv run --python 3.9 --with pyyaml python demo/server.py            # serve on :8400
    curl localhost:8400/version
    curl -s localhost:8400/demo/scenario | python -m json.tool         # K streams + induced recovery
    curl -s localhost:8400/v1/chat/completions \
      -H 'authorization: Bearer sk-cairn-demo' -H 'content-type: application/json' \
      -d '{"model":"gpt-oss-120b","messages":[{"role":"user","content":"hi"}],"max_tokens":12}'
"""

from __future__ import annotations

import json
import pathlib
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scheduler" / "src"))

from cairn_scheduler import fit, load_model_config  # noqa: E402
from cairn_scheduler.gateway import Gateway, GatewayError  # noqa: E402
from cairn_scheduler.recovery import RecoveryManager  # noqa: E402
from cairn_scheduler.runtime import build_mock_pipeline  # noqa: E402
from cairn_scheduler.scheduler import Scheduler, Stream  # noqa: E402
from cairn_scheduler.sim import Sim  # noqa: E402

API_KEY = "sk-cairn-demo"
CFG = ROOT / "configs" / "gpt-oss-120b.yaml"
MAX_BODY_BYTES = 1_048_576  # 1 MiB request-body ceiling (H2)


class Engine:
    """Holds the model fit; runs each request through a fresh deterministic sim."""

    def __init__(self) -> None:
        self.cfg = load_model_config(CFG)
        self.fit = fit(self.cfg, target_k=8, context_len=4096)
        self.gateway = Gateway(
            api_keys={API_KEY},
            model_names={self.cfg.name, "qwen3.5-397b-a17b"},
            default_version="0.1.0",
        )

    def complete(self, req) -> dict:
        sim = Sim()
        sched = Scheduler(sim, build_mock_pipeline(self.fit), vocab_size=self.cfg.vocab_size, k_max=8)
        stream = self.gateway.admit(req)
        sched.submit(stream)
        sim.run()
        return self.gateway.format_response(req, stream)

    def scenario(self, k: int = 6, evict: bool = True) -> dict:
        sim = Sim()
        sched = Scheduler(sim, build_mock_pipeline(self.fit), vocab_size=self.cfg.vocab_size, k_max=k)
        rm = RecoveryManager(sim, sched, warm_spares=1)
        if evict:
            rm.arm(self.fit.n // 2, after_stream="s0", after_count=5)
        sched.submit_all(
            [Stream(id=f"s{i}", prompt=[i + 1, i + 2, i + 3], max_new_tokens=20) for i in range(k)]
        )
        sim.run()
        return {
            "model": self.cfg.name,
            "n_stages": self.fit.n,
            "k_streams": k,
            "completed": len(sched.finished),
            "avg_occupancy": round(sched.avg_occupancy(), 3),
            "recovery": [
                {"stage": e.stage, "policy": e.policy, "replay_tokens": e.replay_tokens,
                 "t_warning": e.t_warning, "t_resume": round(e.t_resume, 2)}
                for e in rm.timeline
            ],
        }


ENGINE: Engine | None = None


def _engine() -> Engine:
    global ENGINE
    if ENGINE is None:
        ENGINE = Engine()
    return ENGINE


class Handler(BaseHTTPRequestHandler):
    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        eng = _engine()
        if self.path == "/version":
            self._send(eng.gateway.version_info())
        elif self.path == "/v1/health":
            self._send({"status": "ok"})
        elif self.path.startswith("/demo/scenario"):
            self._send(eng.scenario())
        else:
            self._send({"error": {"message": "not found", "type": "not_found"}}, 404)

    def _read_capped(self, length: int):
        """Read the body buffering at most MAX_BODY_BYTES, but DRAIN the rest so the client
        still receives our response cleanly (rejecting on Content-Length alone and closing
        mid-upload resets the connection). Returns (kept_bytes, total_bytes_seen)."""
        kept = bytearray()
        total = 0
        remaining = max(0, length)
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 65536))
            if not chunk:
                break
            remaining -= len(chunk)
            total += len(chunk)
            if len(kept) < MAX_BODY_BYTES:
                kept.extend(chunk[: MAX_BODY_BYTES - len(kept)])
        return bytes(kept), total

    def do_POST(self):
        eng = _engine()
        if self.path != "/v1/chat/completions":
            self._send({"error": {"message": "not found", "type": "not_found"}}, 404)
            return
        try:
            eng.gateway.authenticate(self.headers.get("authorization"))
            try:
                length = int(self.headers.get("content-length", 0))
            except ValueError:
                raise GatewayError(400, "invalid Content-Length")
            raw, total = self._read_capped(length)
            if total > MAX_BODY_BYTES:
                raise GatewayError(413, "request body too large")
            raw = raw or b"{}"
            try:
                body = json.loads(raw)
            except json.JSONDecodeError:
                raise GatewayError(400, "invalid JSON body")
            req = eng.gateway.parse_request(body)
            self._send(eng.complete(req))
        except GatewayError as e:
            self._send(e.to_error(), e.status)
        except Exception:  # never leak a traceback to the client (M10)
            self._send({"error": {"message": "internal error", "type": "internal_error"}}, 500)


def make_server(port: int = 8400) -> ThreadingHTTPServer:
    _engine()  # build eagerly (single-threaded) so the lazy init can't race under threads (L8)
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8400
    print(f"Cairn demo on http://127.0.0.1:{port}  (API key: {API_KEY})")
    make_server(port).serve_forever()
