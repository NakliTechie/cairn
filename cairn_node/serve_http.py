"""cairn_node.serve_http — the Cairn distributed-with-recovery BYOK endpoint (runs on rank0 of a fleet).

The OpenAI-compatible front for a LIVE cairn_node fleet. **Embrace + extend:** REUSE the gateway's BYOK
Bearer auth + request validation (`cairn_scheduler.gateway`), REUSE HF/SGLang for the real tokenizer; the
only net-new is driving the token stream across the wire-stitched fleet WITH recovery — a node dies
mid-decode → re-stitch to the warm spare → replay → resume, WHILE serving the request. That recovery is
the thing vLLM/SGLang multi-node serving can't do (their NCCL pipeline aborts on a reclaim).

    # on rank0 of a running fleet (entry local; tail + spare dial back to the driver):
    python -m cairn_node.serve_http --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 --api-key sk-... \\
        --entry-host 127.0.0.1 --entry-port 7777 --tail-sink 7779 --spare-host <ip2> --spare-sink 7780 \\
        --spare-port 7777 --n-stages 2
    curl -s localhost:8000/v1/chat/completions -H 'authorization: Bearer sk-...' \\
        -d '{"model":"Qwen/Qwen2.5-0.5B-Instruct","messages":[{"role":"user","content":"hi"}],"max_tokens":16}'
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "fork"), str(_ROOT / "scheduler" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from cairn_scheduler.gateway import Gateway, GatewayError  # noqa: E402  (reuse: auth + validation + OpenAI shapes)
from shard.transport import LanEdge  # noqa: E402
from shard import wire  # noqa: E402
from cairn_node.serve import _connect_retry, _listen  # noqa: E402
from cairn_node.pipeline import _drive_multi  # noqa: E402
from cairn_node.recovery import decode_with_recovery  # noqa: E402

MAX_BODY_BYTES = 1_048_576


class _Tok:
    """Real tokenizer for serving (HF AutoTokenizer + chat template); a char-based stand-in for `mock:N`
    models so the HTTP/drive plumbing is CPU-testable without downloading weights."""

    def __init__(self, model: str) -> None:
        self.mock = model.startswith("mock:")
        if self.mock:
            self.vocab = int(model.split(":")[-1]) if ":" in model else 256
            self.t = None
        else:
            from transformers import AutoTokenizer
            self.t = AutoTokenizer.from_pretrained(model)

    def encode(self, messages) -> list:
        if self.mock:
            text = " ".join(str(m.get("content", "")) for m in messages)
            return [ord(c) % self.vocab for c in text] or [1]
        ids = self.t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return list(ids)

    def decode(self, ids) -> str:
        if self.mock:
            return "".join(chr(33 + (i % 90)) for i in ids)         # printable stand-in (plumbing only)
        return self.t.decode(ids, skip_special_tokens=True)


def connect_fleet(entry_host: str, entry_port: int, tail_sink: int, bind_host: str = "0.0.0.0",
                  spare_sink: int | None = None):
    """Dial the local entry node + accept the tail (and optionally the warm spare) dialing back. Returns
    (head, tail, spare|None) — the persistent fleet connection the endpoint drives every request over.

    The driver's read edges are SUPERVISED (supervised_recv_timeout=True): an abrupt tail reclaim that
    leaves the TCP connection half-open (no FIN/RST) makes active.recv() raise socket.timeout after the
    generous CAIRN_EDGE_RECV_TIMEOUT deadline instead of blocking forever — so decode_with_recovery
    detects the death and re-stitches to the spare. This is the fix for the live between-requests hang
    (2026-06-22): on localhost a SIGKILL'd peer is reset instantly, but a reclaimed BOX never sends a
    FIN, so without the deadline recv() hangs forever and recovery never fires."""
    wire.key_from_env("SHARD_PSK")
    sink_t = _listen(bind_host, tail_sink)
    sink_s = _listen(bind_host, spare_sink) if spare_sink else None
    # head is supervised too: during recovery the driver SENDS set_next + replay through it; a stalled
    # send to a half-open entry should fail fast (death) rather than wedge the recovery.
    head = LanEdge(entry_host, entry_port, supervised_recv_timeout=True); _connect_retry(head)
    ct, _ = sink_t.accept(); tail = LanEdge.from_socket(ct, supervised_recv_timeout=True)
    spare = None
    if sink_s is not None:
        cs, _ = sink_s.accept(); spare = LanEdge.from_socket(cs, supervised_recv_timeout=True)
    return head, tail, spare


class FleetEngine:
    """Holds the persistent fleet connection + the tokenizer; serves one request at a time over the single
    wire (a lock — v0; concurrent multi-stream is the upgrade via _drive_multi). Survives ONE node death
    per spare via decode_with_recovery (after which the spare becomes the tail; Path 2 replenishes)."""

    def __init__(self, model: str, head, tail, spare=None, spare_host=None, spare_port=None,
                 api_key: str | None = None, log=None) -> None:
        self.model = model
        self.tok = _Tok(model)
        self.head, self.tail, self.spare = head, tail, spare
        self.spare_host, self.spare_port = spare_host, spare_port
        self.gateway = Gateway(api_keys={api_key} if api_key else set(), model_names={model},
                               default_version="0.1.0")
        self._lock = threading.Lock()
        self._n = 0
        # Where recovery progress goes. Default: flushed stdout (the yaml tee's it to /tmp/cairn-serve.log),
        # so a node death / re-stitch / MTTR is VISIBLE on the box mid-request — the observability that
        # turned the silent live hang into a diagnosable one. Must never raise (truthful-run discipline).
        self._log = log if log is not None else (lambda m: print(m, flush=True))

    def _on_event(self, kind, *p):
        """on_event hook handed to decode_with_recovery — writes the recovery steps to the serve log so a
        cross-box run leaves a durable, flushed trace (death detected / recovered / per-token)."""
        try:
            if kind == "death":
                self._log(f"[serve_http] *** TAIL DEATH detected after {p[0]} committed tokens "
                          f"— re-stitching entry -> warm spare {self.spare_host}:{self.spare_port}, replaying")
            elif kind == "recovered":
                self._log(f"[serve_http] *** RECOVERED in {p[0]:.3f}s (MTTR) — resumed at token {p[2]} = {p[1]}; "
                          f"spare is now the tail (out of spares until Path 2)")
        except Exception:
            pass

    def run(self, req):
        """Tokenize → drive the fleet (with recovery if a spare is attached) → return (out_ids, mttr)."""
        ids = self.tok.encode(req.messages)
        with self._lock:                                            # one stream at a time over the single wire
            self._n += 1
            seq = f"http-{self._n}"
            if self.spare is not None:
                out, mttr = decode_with_recovery(self.head, self.tail, self.spare, self.spare_host,
                                                 self.spare_port, ids, req.max_tokens, seq=seq,
                                                 on_event=self._on_event)
                if mttr is not None:                                # a node died + we recovered onto the spare:
                    self.tail, self.spare = self.spare, None        # spare is the tail now; out of spares (Path 2)
            else:
                out = _drive_multi(self.head, self.tail, [(seq, ids)], req.max_tokens, stop=False)[seq]
                mttr = None
            return out, mttr, seq

    def complete(self, req) -> dict:
        out, mttr, seq = self.run(req)
        return self._response(req, seq, out, mttr)

    def _response(self, req, seq, out, mttr) -> dict:
        n_prompt = len(self.tok.encode(req.messages))
        resp = {
            "id": seq, "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.tok.decode(out)},
                         "finish_reason": "length"}],
            "usage": {"prompt_tokens": n_prompt, "completion_tokens": len(out),
                      "total_tokens": n_prompt + len(out)},
        }
        if mttr is not None:                                        # surface that we recovered mid-request
            resp["cairn"] = {"recovered": True, "mttr_s": round(mttr, 3)}
        return resp

    def sse_chunks(self, req, seq, out):
        for i, tok in enumerate(out):
            yield {"id": seq, "object": "chat.completion.chunk", "model": req.model,
                   "choices": [{"index": 0, "delta": {"content": self.tok.decode([tok])}, "finish_reason": None}]}
        yield {"id": seq, "object": "chat.completion.chunk", "model": req.model,
               "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}


class _Handler(BaseHTTPRequestHandler):
    engine: FleetEngine = None  # set by make_server

    def _send(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status); self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body))); self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        if self.path == "/version":
            self._send(self.engine.gateway.version_info())
        elif self.path == "/v1/health":
            self._send({"status": "ok", "model": self.engine.model})
        else:
            self._send({"error": {"message": "not found", "type": "not_found", "code": "not_found"}}, 404)

    def do_POST(self):
        eng = self.engine
        if self.path != "/v1/chat/completions":
            self._send({"error": {"message": "not found", "type": "not_found", "code": "not_found"}}, 404)
            return
        try:
            eng.gateway.authenticate(self.headers.get("authorization"))
            length = int(self.headers.get("content-length", 0) or 0)
            if length > MAX_BODY_BYTES:
                raise GatewayError(413, "request body too large")
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                raise GatewayError(400, "invalid JSON body")
            req = eng.gateway.parse_request(body)
            if req.stream:
                out, mttr, seq = eng.run(req)
                self.send_response(200); self.send_header("content-type", "text/event-stream")
                self.send_header("cache-control", "no-cache"); self.end_headers()
                for chunk in eng.sse_chunks(req, seq, out):
                    self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n"); self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
            else:
                self._send(eng.complete(req))
        except GatewayError as e:
            self._send(e.to_error(), e.status)
        except Exception as e:  # never leak internals to the client
            self._send({"error": {"message": f"internal error: {type(e).__name__}", "type": "internal_error",
                                   "code": "internal_error"}}, 500)


def make_server(engine: FleetEngine, host: str, port: int) -> ThreadingHTTPServer:
    _Handler.engine = engine
    return ThreadingHTTPServer((host, port), _Handler)


def main() -> None:
    ap = argparse.ArgumentParser(description="Cairn distributed-with-recovery OpenAI BYOK endpoint")
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--api-key", default=os.environ.get("CAIRN_API_KEY", ""))
    ap.add_argument("--entry-host", default="127.0.0.1")
    ap.add_argument("--entry-port", type=int, default=7777)
    ap.add_argument("--tail-sink", type=int, default=7779)
    ap.add_argument("--spare-host", default="")               # set to enable warm-spare recovery
    ap.add_argument("--spare-sink", type=int, default=7780)
    ap.add_argument("--spare-port", type=int, default=7777)
    a = ap.parse_args()
    head, tail, spare = connect_fleet(a.entry_host, a.entry_port, a.tail_sink, a.bind,
                                      spare_sink=a.spare_sink if a.spare_host else None)
    eng = FleetEngine(a.model, head, tail, spare=spare,
                      spare_host=a.spare_host or None, spare_port=a.spare_port,
                      api_key=a.api_key or None)
    rec = "ON (warm spare attached)" if spare is not None else "off (no spare)"
    print(f"[serve_http] Cairn endpoint on http://{a.bind}:{a.port}  model={a.model}  recovery={rec}", flush=True)
    make_server(eng, a.bind, a.port).serve_forever()


if __name__ == "__main__":
    main()
