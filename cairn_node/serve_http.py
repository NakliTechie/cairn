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
import select
import subprocess
import sys
import threading
import time
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
from cairn_node.recovery import decode_with_recovery, decode_with_recovery_nstage  # noqa: E402

MAX_BODY_BYTES = 1_048_576

# Optional request heartbeat: if CAIRN_ACTIVITY_FILE is set, stamp it (mtime) on each authed request so an
# external idle-autostop watchdog can tear the fleet down after N minutes of no queries. Off when unset →
# the proven serve path is byte-for-byte unchanged. Must never raise (truthful-run discipline).
_ACTIVITY_FILE = os.environ.get("CAIRN_ACTIVITY_FILE")


def _touch_activity() -> None:
    if not _ACTIVITY_FILE:
        return
    try:
        with open(_ACTIVITY_FILE, "w") as f:
            f.write(str(time.time()))
    except Exception:
        pass


class _Tok:
    """Real tokenizer for serving (HF AutoTokenizer + chat template); a char-based stand-in for `mock:N`
    models so the HTTP/drive plumbing is CPU-testable without downloading weights."""

    def __init__(self, model: str) -> None:
        self.mock = model.startswith("mock:")
        self.eos: set[int] = set()
        if self.mock:
            self.vocab = int(model.split(":")[-1]) if ":" in model else 256
            self.t = None
        else:
            from transformers import AutoTokenizer
            self.t = AutoTokenizer.from_pretrained(model)
            if self.t.eos_token_id is not None:
                self.eos.add(int(self.t.eos_token_id))
            # generation_config may list extra stop ids (DeepSeek ships <｜end▁of▁sentence｜> + variants)
            try:
                from transformers import GenerationConfig
                gc = GenerationConfig.from_pretrained(model)
                ge = getattr(gc, "eos_token_id", None)
                if isinstance(ge, int):
                    self.eos.add(ge)
                elif isinstance(ge, (list, tuple)):
                    self.eos.update(int(x) for x in ge)
            except Exception:
                pass

    def encode(self, messages) -> list:
        if self.mock:
            text = " ".join(str(m.get("content", "")) for m in messages)
            return [ord(c) % self.vocab for c in text] or [1]
        try:
            ids = self.t.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        except Exception:
            # The sgl-project FP8 repack ships a tokenizer_config WITHOUT a chat_template (GPU-confirmed
            # 2026-06-28: apply_chat_template → ValueError), so fall back to the canonical DeepSeek-V3/V4
            # chat format (<｜User｜>/<｜Assistant｜> markers; BOS is added by the tokenizer, add_bos_token=True).
            buf = "".join(m.get("content", "") for m in messages if m.get("role") == "system")
            for m in messages:
                role, content = m.get("role"), m.get("content", "")
                if role == "user":
                    buf += "<｜User｜>" + content
                elif role == "assistant":
                    buf += "<｜Assistant｜>" + content
            buf += "<｜Assistant｜>"
            ids = self.t.encode(buf)
        return list(ids)

    def decode(self, ids) -> str:
        if self.mock:
            return "".join(chr(33 + (i % 90)) for i in ids)         # printable stand-in (plumbing only)
        return self.t.decode(ids, skip_special_tokens=True)


def _accept_spare(sink_s):
    """Accept ONE spare dialing the spare-sink + read its hello to learn its listen addr (for promotion).
    Returns (edge, (host, port)). A spare with no hello (legacy) registers with addr=None — usable only
    for tail-replace (where the driver never needs to dial it)."""
    cs, _ = sink_s.accept()
    edge = LanEdge.from_socket(cs, supervised_recv_timeout=True)
    addr = None
    try:
        # Peek with select BEFORE recv: a real spare announces in <1ms (cs readable → recv the hello). A
        # legacy spare that never announces leaves cs idle → select times out → we DON'T call edge.recv()
        # (which on a supervised edge would treat the timeout as a death and close the socket).
        r, _, _ = select.select([cs], [], [], 1.0)
        if r:
            msg = edge.recv()
            if isinstance(msg, dict) and msg.get("op") == "hello":
                addr = (msg["host"], int(msg["port"]))
    except Exception:
        pass
    return edge, addr


def connect_fleet(entry_host: str, entry_port: int, tail_sink: int, bind_host: str = "0.0.0.0",
                  spare_sink: int | None = None, n_spares: int = 0):
    """Dial the local entry node + accept the tail and the INITIAL warm spares dialing back. Returns
    (head, tail, spares, sink_s) — `spares` is a list of (edge, addr) and `sink_s` is the still-open
    spare-sink listener (kept open so the engine's acceptor thread can register REPLENISHED spares later).

    The driver's read edges are SUPERVISED (supervised_recv_timeout=True): an abrupt reclaim that leaves
    the TCP connection half-open (no FIN/RST) makes recv() raise socket.timeout after CAIRN_EDGE_RECV_TIMEOUT
    instead of blocking forever — so the recovery driver detects the death and re-stitches. (2026-06-22 hang fix.)"""
    wire.key_from_env("SHARD_PSK")
    sink_t = _listen(bind_host, tail_sink)
    sink_s = _listen(bind_host, spare_sink) if spare_sink else None
    head = LanEdge(entry_host, entry_port, supervised_recv_timeout=True); _connect_retry(head)
    ct, _ = sink_t.accept(); tail = LanEdge.from_socket(ct, supervised_recv_timeout=True)
    spares = []
    if sink_s is not None:
        for _ in range(max(0, n_spares)):
            spares.append(_accept_spare(sink_s))
    return head, tail, spares, sink_s


class FleetEngine:
    """Holds the persistent fleet connection + tokenizer; serves one request at a time over the single wire
    (a lock — v0). Survives node deaths via N-stage warm-spare recovery (decode_with_recovery_nstage): ANY
    position (entry/middle/tail) recovers onto a generic spare from the pool; then — if self_replenish — a
    replacement is provisioned in the background to refill the warm pool toward warm_target."""

    def __init__(self, model, head, tail, *, stage_addrs=None, sink_s=None, spares=None,
                 warm_target=1, max_spares=None, self_replenish=False, replenish_cmd=None,
                 api_key=None, log=None,
                 spare=None, spare_host=None, spare_port=None):         # legacy single-spare API (compat)
        self.model = model
        self.tok = _Tok(model)
        self.head, self.tail = head, tail
        self.stage_addrs = list(stage_addrs) if stage_addrs else None   # [(host,port)] per active rank 0..N-1
        self._sink_s = sink_s
        self.spares = list(spares or [])                                # warm pool: [(edge, addr|None)]
        self._legacy_spare_host, self._legacy_spare_port = spare_host, spare_port
        if spare is not None:                                           # legacy: one edge + its addr → pool
            self.spares.append((spare, (spare_host, spare_port) if spare_host else None))
        self.warm_target = warm_target
        self.max_spares = max_spares if max_spares is not None else max(warm_target, 1)
        self.self_replenish = self_replenish
        self.replenish_cmd = replenish_cmd
        self.gateway = Gateway(api_keys={api_key} if api_key else set(), model_names={model},
                               default_version="0.1.0")
        self._lock = threading.Lock()                               # one request at a time over the wire
        self._spares_lock = threading.Lock()                       # guards spares + _provisioning
        self._provisioning = 0                                      # replenishments in flight
        self._n = 0
        # Recovery progress → flushed stdout (yaml tee's it to /tmp/cairn-serve.log): a death / re-stitch /
        # MTTR is VISIBLE on the box mid-request. Must never raise (truthful-run discipline).
        self._log = log if log is not None else (lambda m: print(m, flush=True))
        if self._sink_s is not None:                               # register REPLENISHED spares as they dial in
            threading.Thread(target=self._accept_spares_loop, daemon=True).start()

    def _accept_spares_loop(self):
        """Background: register spares that dial the spare-sink after startup (replenished boxes). Each
        announces its listen addr via a hello (serve.py --announce) so the driver can promote it later."""
        while True:
            try:
                edge, addr = _accept_spare(self._sink_s)
            except OSError:
                return
            with self._spares_lock:
                self.spares.append((edge, addr))
                self._provisioning = max(0, self._provisioning - 1)
            self._log(f"[serve_http] spare registered {addr} — warm pool now {len(self.spares)}")

    def _maybe_replenish(self):
        """After a spare is consumed, launch replacements toward warm_target (capped at max_spares)."""
        if not self.self_replenish or not self.replenish_cmd:
            return
        with self._spares_lock:
            have = len(self.spares) + self._provisioning
            launch = max(0, min(self.warm_target, self.max_spares) - have)
            self._provisioning += launch
        for _ in range(launch):
            threading.Thread(target=self._provision_one, daemon=True).start()

    def _provision_one(self):
        self._log(f"[serve_http] replenish: provisioning a replacement spare → {self.replenish_cmd}")
        try:
            subprocess.run(self.replenish_cmd, shell=True, check=True)   # launches a box that dials spare-sink + announces;
            # the acceptor loop registers it (and decrements _provisioning) when it connects.
        except Exception as e:
            self._log(f"[serve_http] replenish FAILED: {e}")
            with self._spares_lock:
                self._provisioning = max(0, self._provisioning - 1)

    def _on_event(self, kind, *p):
        """Progress hook for the recovery driver — a durable, flushed trace. Handles BOTH the N-stage
        signature ("draining"/"death", stage, committed) and the legacy tail-only one ("draining"/"death",
        committed); the legacy path keeps the original 'TAIL DEATH detected … spare_host:port' wording."""
        try:
            legacy = (kind in ("draining", "death") and len(p) == 1)   # tail-only decode_with_recovery
            tgt = f"{self._legacy_spare_host}:{self._legacy_spare_port}"
            if kind == "draining" and legacy:
                self._log(f"[serve_http] *** PROACTIVE DRAIN signalled after {p[0]} committed tokens "
                          f"— pre-emptively re-stitching entry -> warm spare {tgt}")
            elif kind == "death" and legacy:
                self._log(f"[serve_http] *** TAIL DEATH detected after {p[0]} committed tokens "
                          f"— re-stitching entry -> warm spare {tgt}, replaying")
            elif kind == "draining":
                self._log(f"[serve_http] *** PROACTIVE DRAIN at stage {p[0]} after {p[1]} committed tokens "
                          f"— re-stitching onto a warm spare")
            elif kind == "death":
                self._log(f"[serve_http] *** DEATH at stage {p[0]} after {p[1]} committed tokens "
                          f"— re-stitching onto a warm spare, replaying")
            elif kind == "recovered":
                self._log(f"[serve_http] *** RECOVERED in {p[0]:.3f}s (MTTR) — resumed at token {p[2]} = {p[1]}")
        except Exception:
            pass

    def run(self, req):
        """Tokenize → drive the fleet (any-position recovery if a spare is in the pool) → (out, mttr, seq)."""
        ids = self.tok.encode(req.messages)
        with self._lock:                                            # one stream at a time over the single wire
            self._n += 1
            seq = f"http-{self._n}"
            with self._spares_lock:
                spare_entry = self.spares[0] if self.spares else None
            if spare_entry is not None and self.stage_addrs is not None:
                spare_edge, spare_addr = spare_entry
                out, mttr, new_head, new_read, k = decode_with_recovery_nstage(
                    self.head, self.tail, spare_edge, stage_addrs=self.stage_addrs, spare_addr=spare_addr,
                    prompt=ids, n_new=req.max_tokens, seq=seq, on_event=self._on_event)
                if mttr is not None:                                # consumed a spare; the fleet is healed
                    self.head, self.tail = new_head, new_read
                    with self._spares_lock:
                        if self.spares and self.spares[0] is spare_entry:
                            self.spares.pop(0)
                    if k is not None and spare_addr is not None and 0 <= k < len(self.stage_addrs):
                        self.stage_addrs[k] = spare_addr            # the promoted spare now serves rank k
                    self._maybe_replenish()
            elif spare_entry is not None:                           # legacy tail-only path (no stage_addrs)
                spare_edge, spare_addr = spare_entry
                sh = spare_addr[0] if spare_addr else None
                sp = spare_addr[1] if spare_addr else None
                out, mttr = decode_with_recovery(self.head, self.tail, spare_edge, sh, sp,
                                                 ids, req.max_tokens, seq=seq, on_event=self._on_event)
                if mttr is not None:
                    with self._spares_lock:
                        if self.spares and self.spares[0] is spare_entry:
                            self.spares.pop(0)
                    self.tail = spare_edge
                    self._maybe_replenish()
            else:                                                  # no spare — plain decode
                out = _drive_multi(self.head, self.tail, [(seq, ids)], req.max_tokens, stop=False)[seq]
                mttr = None
            return self._trim_eos(out), mttr, seq

    def _trim_eos(self, out):
        """The decode loop runs a fixed max_tokens with no early-stop, so an instruct model emits EOS at
        its natural end and then DEGENERATES (greedy temp=0 → repetition). Truncate at the first stop id so
        the response ends where the model meant to. (GPU-confirmed 2026-06-28: clean first sentence then
        'PL PL...' to the cap.)"""
        eos = getattr(self.tok, "eos", set())
        if eos:
            for i, t in enumerate(out):
                if t in eos:
                    return out[:i]
        return out

    def complete(self, req) -> dict:
        out, mttr, seq = self.run(req)
        return self._response(req, seq, out, mttr)

    def _response(self, req, seq, out, mttr) -> dict:
        n_prompt = len(self.tok.encode(req.messages))
        resp = {
            "id": seq, "object": "chat.completion", "model": req.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": self.tok.decode(out)},
                         "finish_reason": ("length" if len(out) >= req.max_tokens else "stop")}],
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
               "choices": [{"index": 0, "delta": {}, "finish_reason": ("length" if len(out) >= req.max_tokens else "stop")}]}


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
            _touch_activity()                                       # stamp for the idle-autostop watchdog
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
        except Exception as e:  # never leak internals to the client — but DO surface them server-side
            import traceback
            print(f"[serve_http] request failed: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
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
    ap.add_argument("--spare-host", default="")               # legacy 2-stage tail-only path
    ap.add_argument("--spare-sink", type=int, default=7780)   # the driver's spare-sink port (spares dial here)
    ap.add_argument("--spare-port", type=int, default=7777)   # a spare's listen port (legacy default)
    # N-STAGE recovery: the addresses of the N ACTIVE stages (rank 0..N-1), so the driver can re-stitch
    # ANY position onto a generic spare (fills the promoted spare's new downstream k+1). Spares announce
    # their own listen addr via a hello (serve.py --announce) — so replenished boxes with new IPs work.
    ap.add_argument("--stage-hosts", default="",
                    help="comma-separated host/IP of each active stage rank 0..N-1 (enables N-stage recovery)")
    ap.add_argument("--stage-port", type=int, default=7777, help="listen port shared by the active stages")
    ap.add_argument("--initial-spares", type=int, default=-1,
                    help="how many warm spares dial in at startup (default: 1 if recovery enabled, else 0)")
    ap.add_argument("--warm-target", type=int, default=1, help="warm spares to keep at all times")
    ap.add_argument("--max-spares", type=int, default=0, help="ceiling on warm spares (0 = warm_target)")
    ap.add_argument("--self-replenish", action="store_true",
                    help="after a spare is consumed, provision a replacement toward warm_target")
    ap.add_argument("--replenish-cmd", default=os.environ.get("CAIRN_REPLENISH_CMD", ""),
                    help="shell command that launches one replacement spare (it must dial the spare-sink + "
                         "--announce its addr). e.g. a scoped `sky exec`/`sky launch`. Driver-box creds.")
    a = ap.parse_args()

    use_recovery = bool(a.stage_hosts) or bool(a.spare_host)
    n_init = a.initial_spares if a.initial_spares >= 0 else (1 if use_recovery else 0)
    sink_port = a.spare_sink if use_recovery else None
    head, tail, spares, sink_s = connect_fleet(a.entry_host, a.entry_port, a.tail_sink, a.bind,
                                               spare_sink=sink_port, n_spares=n_init)
    # legacy spare that didn't --announce → fall back to the CLI-provided addr
    if a.spare_host and not a.stage_hosts and spares and spares[0][1] is None:
        spares[0] = (spares[0][0], (a.spare_host, a.spare_port))

    stage_addrs = None
    if a.stage_hosts:
        stage_addrs = [(h.strip(), a.stage_port) for h in a.stage_hosts.split(",") if h.strip()]

    eng = FleetEngine(a.model, head, tail, stage_addrs=stage_addrs, sink_s=sink_s, spares=spares,
                      warm_target=a.warm_target, max_spares=(a.max_spares or None),
                      self_replenish=a.self_replenish, replenish_cmd=(a.replenish_cmd or None),
                      api_key=a.api_key or None)
    if stage_addrs is not None:
        rec = (f"ON N-stage (N={len(stage_addrs)}, warm pool={len(spares)}, warm_target={a.warm_target}, "
               f"replenish={'on' if a.self_replenish else 'off'})")
    elif spares:
        rec = "ON tail-only (warm spare attached)"
    else:
        rec = "off (no spare)"
    print(f"[serve_http] Cairn endpoint on http://{a.bind}:{a.port}  model={a.model}  recovery={rec}", flush=True)
    make_server(eng, a.bind, a.port).serve_forever()


if __name__ == "__main__":
    main()
