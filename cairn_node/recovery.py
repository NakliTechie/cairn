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
                         prompt: List[int], n_new: int, seq: str = "s0"):
    """Greedy-decode `n_new` tokens through head -> … -> active. DETECT a node death (active.recv raises)
    and recover onto the pre-warmed `spare`: re-point the entry (set_next), replay the committed history
    (its last logit is the token we were waiting for — KV rebuild + resume in one prefill), continue.
    Returns (tokens, mttr_s | None). The death is induced externally (CAIRN_DIE_AFTER on the node)."""
    import torch
    from shard.transport import EDGE_ERRORS
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
            t0 = time.time()
            head.send({"op": "set_next", "host": spare_host, "port": spare_port})   # entry -> warm spare
            active = spare                                                          # read from the spare now
            hist = list(prompt) + out                                              # replay rebuilds the spare's KV;
            head.send({"h": torch.tensor([hist]), "seq": seq, "pos": 0})           # last logit = the pending token
            tok = _next_tok(active.recv())
            out.append(tok)
            pos = len(hist)
            cur = torch.tensor([[tok]])
            mttr = time.time() - t0
            recovered = True
            continue
        out.append(tok)
        pos += cur.shape[1]
        cur = torch.tensor([[tok]])
    return out[:n_new], mttr


def _spawn(runtime, model, ls, le, lp, nh, np_, device="cpu", die_after=None):
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    if die_after:
        env["CAIRN_DIE_AFTER"] = str(die_after)          # node hard-exits after this many forwards
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", runtime, "--model", model,
         "--layer-start", str(ls), "--layer-end", str(le), "--device", device,
         "--listen-port", str(lp), "--bind-host", "127.0.0.1", "--next-host", nh, "--next-port", str(np_)],
        cwd=str(_ROOT), env=env)


def _selftest() -> None:
    from shard.transport import LanEdge
    from shard import wire
    os.environ.setdefault("SHARD_PSK", "cairn-dev-psk")
    wire.key_from_env("SHARD_PSK")
    from cairn_node.pipeline import run_pipeline

    NL, CUT, PROMPT, NNEW = 8, 4, [1, 5, 9, 3, 7, 2], 6
    ref = run_pipeline("mock", "mock:8", NL, [CUT], PROMPT, NNEW, base_port=7950)
    print("ref (no kill) :", ref)

    # entry 0..CUT (7911 -> tail 7912); tail CUT..NL DIES after 3 fwds (7912 -> driver tail-sink 7914);
    # spare CUT..NL (7913 -> driver spare-sink 7915). Separate sinks disambiguate tail vs spare.
    entry = _spawn("mock", "mock:8", 0, CUT, 7911, "127.0.0.1", 7912)
    tail  = _spawn("mock", "mock:8", CUT, NL, 7912, "127.0.0.1", 7914, die_after=3)
    spare = _spawn("mock", "mock:8", CUT, NL, 7913, "127.0.0.1", 7915)
    procs = [entry, tail, spare]
    sink_t = _listen("127.0.0.1", 7914)
    sink_s = _listen("127.0.0.1", 7915)
    try:
        head = LanEdge("127.0.0.1", 7911); _connect_retry(head)            # driver -> entry
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct)          # tail  -> driver
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs)         # spare -> driver (standby)
        rec, mttr = decode_with_recovery(head, tail_e, spare_e, "127.0.0.1", 7913, PROMPT, NNEW)
        print("rec (die@3)   :", rec, "MATCH", rec == ref, "| MTTR %.3fs" % (mttr or 0))
        try:
            head.send({"op": "stop"})
        except Exception:
            pass
        assert rec == ref, "recovered output != no-kill reference — recovery bug"
        print(">>> RECOVERY OK (mock node death -> re-stitch to warm spare -> replay -> resume == no-kill)")
    finally:
        for s in (sink_t, sink_s):
            try:
                s.close()
            except Exception:
                pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


if __name__ == "__main__":
    _selftest()
