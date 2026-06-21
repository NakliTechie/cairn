"""cairn_node.pipeline — drive a multi-process Cairn pipeline + greedy-decode (the multi-node counterpart
of test_sglang_split's `_drive`). Spawns one `cairn_node.serve` process per stage, wires them
driver → node0 → … → node(N-1) → driver with LanEdge, feeds a prompt, samples greedily.

Used by scheduler/tests/test_multinode_pipeline.py. Runnable directly as a CPU mock self-test (no GPU):

    SHARD_PSK=dev python -m cairn_node.pipeline
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


def run_pipeline(runtime: str, model: str, num_layers: int, cuts: List[int], prompt: List[int],
                 n_new: int, device: str = "cpu", base_port: int = 7100,
                 python_exe: str = "") -> List[int]:
    """Spawn a stage-per-process pipeline (cut at `cuts`), greedy-decode `n_new` tokens, return them.
    Stages are torn down on exit. Distinct `base_port` per concurrent call (avoids port clashes)."""
    import torch
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
        lsock = _listen(sink)                                # driver's sink — the tail connects here
        head = LanEdge("127.0.0.1", ports[0])
        _connect_retry(head)                                 # driver -> node0 (entry)
        conn, _ = lsock.accept()
        tail = LanEdge.from_socket(conn)                     # node(N-1) -> driver

        cur = torch.tensor([list(prompt)])                   # [1, S] token ids
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


if __name__ == "__main__":
    _selftest()
