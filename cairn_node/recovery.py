"""cairn_node.recovery — driver-side warm-spare recovery (Phase 2, Path 1).

The driver is already in the decode loop AND holds the token history, so it can orchestrate recovery for
a stream with NO control plane: when a node dies (the edge raises) it (1) re-points the entry node at a
PRE-WARMED spare (serve.py `set_next`), (2) replays the committed history through entry->spare so the
spare rebuilds its KV (the replay's last logit IS the next token — rebuild + resume in one prefill),
(3) resumes. Proven here on the CPU mock (a node hard-exits mid-stream via CAIRN_DIE_AFTER, the output
must match the no-kill run); sglang-validated live on a 3+1-box fleet.

    SHARD_PSK=dev python -m cairn_node.recovery
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time
from typing import List

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "fork") not in sys.path:
    sys.path.insert(0, str(_ROOT / "fork"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cairn_node.serve import _connect_retry, _listen  # noqa: E402


def _next_tok(msg) -> int:
    return int(msg["h"][:, -1].argmax(-1))


def decode_with_recovery(head, active, spare, spare_host: str, spare_port: int,
                         prompt: List[int], n_new: int, seq: str = "s0", on_event=None):
    """Greedy-decode `n_new` tokens through head -> … -> active. DETECT a node death (active.recv raises)
    and recover onto the pre-warmed `spare`: re-point the entry (set_next), replay the committed history
    (its last logit is the token we were waiting for — KV rebuild + resume in one prefill), continue.
    Returns (tokens, mttr_s | None). The death is induced externally (CAIRN_DIE_AFTER on the node).

    `on_event(kind, *payload)` — optional progress hook so a live run leaves a durable per-step trace
    ("tok", i, tok) / ("death", n_committed) / ("recovered", mttr_s, tok, i). It must never raise; a
    logging failure cannot be allowed to break the decode (truthful-run discipline)."""
    import torch
    from shard.transport import EDGE_ERRORS

    def _emit(kind, *payload):
        if on_event is not None:
            try:
                on_event(kind, *payload)
            except Exception:
                pass

    cur = torch.tensor([list(prompt)])
    out: List[int] = []
    pos = 0
    mttr = None
    recovered = False
    while len(out) < n_new:
        head.send({"h": cur, "seq": seq, "pos": pos})
        try:
            tok = _next_tok(active.recv())
        except EDGE_ERRORS:
            if recovered:
                raise RuntimeError("second node death — out of warm spares (Path-1 single-spare scope)")
            _emit("death", len(out))                                                # committed tokens before death
            t0 = time.time()
            head.send({"op": "set_next", "host": spare_host, "port": spare_port})   # entry -> warm spare
            active = spare                                                          # read from the spare now
            # FRESH seq id for the rebuild: the surviving entry still holds KV under the OLD seq, so a same-seq
            # replay makes sglang DECODE one token (shape error: [1,S,-1] over one token's worth) instead of
            # PREFILL the history. Under a new seq the entry + the fresh spare both prefill the full history
            # cleanly — KV rebuilt, last logit = the pending token. (The orphaned old-seq KV on the entry is
            # fine for Path-1 single-spare scope; Path 2 frees it. The stateless mock is unaffected.)
            seq = seq + "~r"
            hist = list(prompt) + out                                              # replay rebuilds KV on entry+spare;
            head.send({"h": torch.tensor([hist]), "seq": seq, "pos": 0})           # last logit = the pending token
            tok = _next_tok(active.recv())
            out.append(tok)
            pos = len(hist)
            cur = torch.tensor([[tok]])
            mttr = time.time() - t0
            recovered = True
            _emit("recovered", mttr, tok, len(out))
            continue
        out.append(tok)
        pos += cur.shape[1]
        cur = torch.tensor([[tok]])
        _emit("tok", len(out), tok)
    return out[:n_new], mttr


def _spawn(runtime, model, ls, le, lp, nh, np_, device="cpu", die_after=None, mem_fraction=None):
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    env.pop("CAIRN_DIE_AFTER", None)                      # NEVER inherit an ambient death (the yaml exports it
    #                                                      for the live tail) — only the node we explicitly mark
    #                                                      below dies; else the no-death ref + entry + spare ALL
    #                                                      hard-exit too (the silent 'peer closed' seen on the box).
    if die_after:
        env["CAIRN_DIE_AFTER"] = str(die_after)          # node hard-exits after this many forwards
    if mem_fraction:
        env["CAIRN_SGLANG_MEM_FRACTION"] = str(mem_fraction)  # share one GPU across co-located nodes (no OOM)
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", runtime, "--model", model,
         "--layer-start", str(ls), "--layer-end", str(le), "--device", device,
         "--listen-port", str(lp), "--bind-host", "127.0.0.1", "--next-host", nh, "--next-port", str(np_)],
        cwd=str(_ROOT), env=env)


def prove_recovery(runtime: str, model: str, num_layers: int, cut: int, prompt: List[int], n_new: int,
                   *, die_after: int, device: str = "cpu", mem_fraction=None, base_port: int = 7900,
                   log_path: str | None = None):
    """Single-box live recovery proof (all nodes on localhost). Build the no-death REFERENCE (entry+tail,
    no spare), then stand up entry / tail / spare, kill the tail after `die_after` forwards, recover onto
    the warm spare (set_next -> replay -> resume), and assert recovered == reference. Runtime-agnostic:
    `mock` on CPU (the no-spend self-test) OR `sglang` on ONE GPU — the CHEAP-FIRST proof that the real
    sglang set_next/replay path works live, before paying for the cross-box run. Returns (ref, rec, mttr).

    `log_path` writes a flushed+fsync'd verdict on the box (survives a kill — truthful-run discipline)."""
    from shard.transport import LanEdge
    from shard import wire
    os.environ.setdefault("SHARD_PSK", "cairn-dev-psk")
    wire.key_from_env("SHARD_PSK")
    from cairn_node.pipeline import run_pipeline

    _fh = open(log_path, "a", buffering=1) if log_path else None

    def log(msg: str) -> None:
        print(msg, flush=True)
        if _fh is not None:
            _fh.write(msg + "\n"); _fh.flush(); os.fsync(_fh.fileno())

    log(f"[prove] runtime={runtime} model={model} layers={num_layers} cut={cut} "
        f"die_after={die_after} device={device} prompt={prompt} n_new={n_new}")

    # 1) no-death REFERENCE — same 2-stage split, spawned + torn down by run_pipeline.
    ref = run_pipeline(runtime, model, num_layers, [cut], prompt, n_new, device=device, base_port=base_port)
    log(f"[prove] ref (no death) = {ref}")

    # 2) entry(0..cut) / tail(cut..NL, dies@die_after) / spare(cut..NL, standby) on localhost; driver recovers.
    b = base_port + 10
    pE, pT, pSpare, sinkT, sinkS = b, b + 1, b + 2, b + 3, b + 4
    entry = _spawn(runtime, model, 0, cut, pE, "127.0.0.1", pT, device=device, mem_fraction=mem_fraction)
    tail  = _spawn(runtime, model, cut, num_layers, pT, "127.0.0.1", sinkT, device=device,
                   die_after=die_after, mem_fraction=mem_fraction)
    spare = _spawn(runtime, model, cut, num_layers, pSpare, "127.0.0.1", sinkS, device=device,
                   mem_fraction=mem_fraction)
    procs = [entry, tail, spare]
    sink_t = _listen("127.0.0.1", sinkT)
    sink_s = _listen("127.0.0.1", sinkS)
    try:
        head = LanEdge("127.0.0.1", pE); _connect_retry(head)              # driver -> entry
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)          # tail  -> driver
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)         # spare -> driver (standby)

        def _on(kind, *p):
            if kind == "tok":
                log(f"[prove] tok {p[0]}/{n_new} = {p[1]}")
            elif kind == "death":
                log(f"[prove] *** NODE DEATH after {p[0]} committed tokens — re-stitch entry -> spare, replay")
            elif kind == "recovered":
                log(f"[prove] *** RECOVERED in {p[0]:.3f}s — resumed at tok {p[2]} = {p[1]}")

        rec, mttr = decode_with_recovery(head, tail_e, spare_e, "127.0.0.1", pSpare, prompt, n_new, on_event=_on)
        log("[prove] rec (die@%d) = %s  MATCH=%s  MTTR=%s"
            % (die_after, rec, rec == ref, ("%.3fs" % mttr) if mttr is not None else "none"))
        try:
            head.send({"op": "stop"})
        except Exception:
            pass
        if rec != ref:
            log(">>> RECOVERY FAIL — recovered output != no-death reference")
            raise AssertionError(f"recovered {rec} != reference {ref}")
        log(">>> RECOVERY OK (node death -> re-stitch to warm spare -> replay -> resume == no-death)")
        return ref, rec, mttr
    finally:
        for s in (sink_t, sink_s):
            try:
                s.close()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
        if _fh is not None:
            _fh.close()


def _selftest() -> None:
    """No-spend mock proof (CPU). The runtime-agnostic core is prove_recovery; sglang runs it on one GPU."""
    prove_recovery("mock", "mock:8", 8, 4, [1, 5, 9, 3, 7, 2], 6, die_after=3, base_port=7950)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="single-box live recovery proof (mock on CPU / sglang on one GPU)")
    ap.add_argument("--runtime", default="mock", choices=["mock", "sglang"])
    ap.add_argument("--model", default="mock:8")
    ap.add_argument("--num-layers", type=int, default=8)
    ap.add_argument("--cut", type=int, default=4)
    ap.add_argument("--prompt", default="1,5,9,3,7,2")
    ap.add_argument("--n-new", type=int, default=6)
    ap.add_argument("--die-after", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--mem-fraction", default=None)       # CAIRN_SGLANG_MEM_FRACTION per node (one-GPU co-tenancy)
    ap.add_argument("--base-port", type=int, default=7900)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    prompt = [int(x) for x in a.prompt.split(",") if x.strip()]
    prove_recovery(a.runtime, a.model, a.num_layers, a.cut, prompt, a.n_new, die_after=a.die_after,
                   device=a.device, mem_fraction=a.mem_fraction, base_port=a.base_port, log_path=a.log)


if __name__ == "__main__":
    main()
