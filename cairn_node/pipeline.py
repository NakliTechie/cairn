"""cairn_node.pipeline — drive a multi-process Cairn pipeline + greedy-decode (the multi-node counterpart
of test_sglang_split's `_drive`). Two drivers share one decode loop (`_drive`):

  • run_pipeline — spawns one `cairn_node.serve` per stage ON ONE HOST (localhost wire), drives, tears down.
  • run_remote   — drives ALREADY-RUNNING nodes launched separately on their own boxes (the cross-box
                   fleet over the VPC LAN). Nodes are NOT spawned here; the driver just connects + drives.

Used by scheduler/tests/test_multinode_pipeline.py. Runnable directly:

    SHARD_PSK=dev python -m cairn_node.pipeline                # CPU mock split==unsplit self-test (no GPU)
    python -m cairn_node.pipeline remote --head-port 7777 --sink-port 7779   # drive a running fleet
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
from typing import List

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "fork") not in sys.path:
    sys.path.insert(0, str(_ROOT / "fork"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from cairn_node.serve import _connect_retry, _listen  # noqa: E402


def _drive(head, tail, prompt: List[int], n_new: int) -> List[int]:
    """Greedy-decode `n_new` tokens over an already-connected pipeline: send running token(s) to the head
    (entry node), recv logits from the tail, argmax, repeat. Host-agnostic — the SAME loop backs the
    local-spawn driver (run_pipeline) and the cross-box fleet driver (run_remote)."""
    import torch
    cur = torch.tensor([list(prompt)])                       # [1, S] token ids
    out: List[int] = []
    pos = 0
    for _ in range(n_new):
        s_len = cur.shape[1]
        head.send({"h": cur, "seq": "s0", "pos": pos})
        msg = tail.recv()
        nxt = int(msg["h"][:, -1].argmax(-1))
        out.append(nxt)
        cur = torch.tensor([[nxt]])
        pos += s_len
    try:
        head.send({"op": "stop"})
    except Exception:
        pass
    return out


def _drive_multi(head, tail, streams, n_new: int, stop: bool = True):
    """Greedy-decode K CONCURRENT streams over an already-connected pipeline — the multi-stream counterpart
    of _drive (the Chunk C v1.1 harness). Each stream has its own `seq` + token history; the driver keeps
    exactly ONE forward per stream in flight, so with K≈N streams all N stages stay busy at once (the
    occupancy gate). Each returned logit carries its `seq`, so the driver routes it to the right stream
    (the nodes keep per-seq paged KV, so streams never collide — the cross-stream KV-isolation gate).

    `streams` = [(seq, prompt), …]. Returns {seq: [tokens]}. `stop=False` leaves the pipeline up for another
    round over the same connection (the bench driver runs multi + baseline + per-stream check rounds, then
    stops once). (Assumes a modest K — the K prefills go up front; KB-scale hidden states fit the buffers.)"""
    import torch
    st = {seq: {"cur": torch.tensor([list(prompt)]), "out": [], "pos": 0} for seq, prompt in streams}
    for seq in st:                                            # prime: one prefill per stream enters the pipeline
        head.send({"h": st[seq]["cur"], "seq": seq, "pos": 0})
    live = set(st)
    while live:
        msg = tail.recv()                                    # next finished forward (carries its seq)
        seq = msg["seq"]
        s = st[seq]
        s["pos"] += s["cur"].shape[1]
        s["out"].append(int(msg["h"][:, -1].argmax(-1)))
        if len(s["out"]) >= n_new:
            live.discard(seq)
            continue
        s["cur"] = torch.tensor([[s["out"][-1]]])
        head.send({"h": s["cur"], "seq": seq, "pos": s["pos"]})   # re-inject → keep this stream in flight
    if stop:
        try:
            head.send({"op": "stop"})
        except Exception:
            pass
    return {seq: s["out"][:n_new] for seq, s in st.items()}


def run_pipeline(runtime: str, model: str, num_layers: int, cuts: List[int], prompt: List[int],
                 n_new: int, device: str = "cpu", base_port: int = 7100,
                 python_exe: str = "", streams=None):
    """Spawn a stage-per-process pipeline (cut at `cuts`) ON ONE HOST (localhost wire), greedy-decode
    `n_new` tokens, return them. Stages are torn down on exit. Distinct `base_port` per concurrent call.
    The cross-box counterpart is run_remote (nodes launched separately, driver only connects).
    If `streams` (list of (seq, prompt)) is given, drives K CONCURRENT streams and returns {seq: [tokens]}
    (the v1.1 multi-stream harness); otherwise single-stream, returns [tokens]."""
    from shard.transport import LanEdge
    from shard import wire

    os.environ.setdefault("SHARD_PSK", "cairn-dev-psk")      # consistent key for driver + node procs
    wire.key_from_env("SHARD_PSK")                           # the driver seals/opens wire frames too

    python_exe = python_exe or sys.executable
    bnd = [0, *cuts, num_layers]                              # contiguous layer boundaries
    nstages = len(bnd) - 1
    ports = [base_port + i for i in range(nstages)]
    sink = base_port + nstages                               # the driver's own listen port (tail dials it)
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}           # SHARD_PSK propagates to the nodes
    env.pop("CAIRN_DIE_AFTER", None)                         # no-death reference driver: don't inherit an ambient death
    # Force the sglang KV-pool fraction into the node env (don't rely on inheritance — the multinode OOM
    # was nodes falling back to the 0.8 default). 0.2 fits two ~6 GiB nodes on one 22 GiB L4; a true
    # 2-GPU run (one node per GPU, own box) can set CAIRN_SGLANG_MEM_FRACTION higher.
    env["CAIRN_SGLANG_MEM_FRACTION"] = os.environ.get("CAIRN_SGLANG_MEM_FRACTION", "0.2")

    procs = []
    for i in range(nstages):
        nxt_port = ports[i + 1] if i < nstages - 1 else sink
        procs.append(subprocess.Popen(
            [python_exe, "-m", "cairn_node.serve", "--runtime", runtime, "--model", model,
             "--layer-start", str(bnd[i]), "--layer-end", str(bnd[i + 1]), "--device", device,
             "--listen-port", str(ports[i]), "--next-host", "127.0.0.1", "--next-port", str(nxt_port)],
            cwd=str(_ROOT), env=env))

    head = tail = lsock = None
    try:
        lsock = _listen("127.0.0.1", sink)                   # driver's sink — the tail connects here
        head = LanEdge("127.0.0.1", ports[0])
        _connect_retry(head)                                 # driver -> node0 (entry)
        conn, _ = lsock.accept()
        tail = LanEdge.from_socket(conn)                     # node(N-1) -> driver
        if streams is not None:
            return _drive_multi(head, tail, streams, n_new)
        return _drive(head, tail, prompt, n_new)
    finally:
        for e in (head, tail):
            if e is not None:
                e.close()
        if lsock is not None:
            lsock.close()
        for p in procs:
            try:
                p.wait(timeout=30)
            except Exception:
                p.kill()


def run_remote(head_host: str, head_port: int, sink_port: int, prompt: List[int], n_new: int,
               bind_host: str = "0.0.0.0", streams=None):
    """Drive a pipeline of ALREADY-RUNNING remote nodes — the cross-box fleet. Unlike run_pipeline this
    does NOT spawn: each node runs on its own box (cairn_node.serve --bind-host <private IP>). The driver
    is co-located with the entry node (rank 0): it dials the head and listens on `sink_port` for the tail
    to dial back. Topology: head(rank0) -> … -> tail -> driver(rank0:sink_port).
    If `streams` (list of (seq, prompt)) is given, drives K CONCURRENT streams → {seq: [tokens]} (v1.1)."""
    from shard.transport import LanEdge
    from shard import wire

    os.environ.setdefault("SHARD_PSK", "cairn-dev-psk")
    wire.key_from_env("SHARD_PSK")

    head = tail = lsock = None
    try:
        lsock = _listen(bind_host, sink_port)                # the tail (the other box) dials this
        head = LanEdge(head_host, head_port)
        _connect_retry(head)                                 # driver -> entry node (localhost on rank 0)
        conn, _ = lsock.accept()
        tail = LanEdge.from_socket(conn)
        if streams is not None:
            return _drive_multi(head, tail, streams, n_new)
        return _drive(head, tail, prompt, n_new)
    finally:
        for e in (head, tail):
            if e is not None:
                e.close()
        if lsock is not None:
            lsock.close()


def _selftest() -> None:
    prompt, n_new, n_layers = [1, 5, 9, 3, 7, 2], 6, 8
    ref = run_pipeline("mock", "mock:8", n_layers, [], prompt, n_new, base_port=7300)
    print("ref  (unsplit):", ref)
    s2 = run_pipeline("mock", "mock:8", n_layers, [4], prompt, n_new, base_port=7400)
    print("split [4]     :", s2, "MATCH", ref == s2)
    s3 = run_pipeline("mock", "mock:8", n_layers, [3, 6], prompt, n_new, base_port=7500)
    print("split [3,6]   :", s3, "MATCH", ref == s3)
    assert ref == s2 == s3, "multi-process split != unsplit — plumbing bug"
    print(">>> PIPELINE PLUMBING OK (multi-process split == unsplit over the wire)")


def _main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="cairn_node pipeline driver")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="CPU mock split==unsplit self-test (default)")
    rp = sub.add_parser("remote", help="drive already-running remote nodes (the cross-box fleet)")
    rp.add_argument("--head-host", default="127.0.0.1")      # entry node (localhost when driver is on rank 0)
    rp.add_argument("--head-port", type=int, required=True)
    rp.add_argument("--sink-port", type=int, required=True)  # the tail dials here
    rp.add_argument("--bind-host", default="0.0.0.0")        # driver sink bind (routable for the tail's box)
    rp.add_argument("--prompt", default="1,5,9,3,7,2")
    rp.add_argument("--n-new", type=int, default=8)
    a = ap.parse_args()
    if a.cmd == "remote":
        toks = [int(x) for x in a.prompt.split(",") if x.strip()]
        print("TOKENS", run_remote(a.head_host, a.head_port, a.sink_port, toks, a.n_new, bind_host=a.bind_host))
    else:
        _selftest()


if __name__ == "__main__":
    _main()
