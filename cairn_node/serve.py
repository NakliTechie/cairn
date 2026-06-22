"""cairn_node.serve — ONE pipeline node: load a block via a NodeRuntime, then run the wire loop —
recv {h, seq, pos} from the previous node (or the driver, for the entry node), `forward`, send the
result to the next node (or back to the driver, for the tail). Separate process per node, so each
sglang ModelRunner gets its own process (the global TP group can't be init'd twice — found 2026-06-21).

    python -m cairn_node.serve --runtime sglang --model Qwen/Qwen2.5-0.5B-Instruct \
        --layer-start 0 --layer-end 12 --device cuda:0 \
        --listen-port 7000 --next-host 127.0.0.1 --next-port 7001

Topology is on argv (v1.0 minimal — no control-plane yet). Each node LISTENS first, then dials its
next (retry), then accepts its prev — so all listeners are up before any dial, no ordering deadlock.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import select
import socket
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "fork") not in sys.path:
    sys.path.insert(0, str(_ROOT / "fork"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _listen(host: str, port: int) -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))                  # 127.0.0.1 single-box; 0.0.0.0 / private IP for the cross-box fleet
    s.listen(1)
    return s


def _connect_retry(edge, tries: int = 600, delay: float = 0.1) -> None:
    for _ in range(tries):
        try:
            edge.connect()
            return
        except OSError:
            time.sleep(delay)
    raise SystemExit(f"[serve] could not connect to {edge.peer_host}:{edge.peer_port}")


def _load_serialized(rt, device: str) -> None:
    """Run load_shard under a per-GPU file lock. sglang's init runs a memory-profiling forward whose
    peak collides and OOMs when two nodes load CONCURRENTLY on one GPU (the dev 2-on-1-L4 case). The
    lock serializes loads on the SAME device; nodes on DIFFERENT GPUs (the real 2-GPU run) still load
    in parallel. CPU/mock loads are instant — the lock is uncontended there."""
    import fcntl
    safe = device.replace(":", "-").replace("/", "-")
    with open(f"/tmp/cairn-load-{safe}.lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            rt.load_shard()
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _runtime_cls(name: str):
    if name == "sglang":
        from shard.sglang_node import SglangNodeRuntime
        return SglangNodeRuntime
    if name == "mock":
        from cairn_node._mock import MockNodeRuntime
        return MockNodeRuntime
    raise SystemExit(f"[serve] unknown runtime {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime", default="sglang", choices=["sglang", "mock"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer-start", type=int, required=True)
    ap.add_argument("--layer-end", type=int, required=True)   # EXCLUSIVE (LayerRange convention)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--listen-port", type=int, required=True)
    ap.add_argument("--bind-host", default="127.0.0.1")      # 0.0.0.0 / private IP for the cross-box fleet
    ap.add_argument("--next-host", default="127.0.0.1")
    ap.add_argument("--next-port", type=int, required=True)
    a = ap.parse_args()

    from shard.node import LayerRange
    from shard.transport import LanEdge
    from shard import wire
    wire.key_from_env("SHARD_PSK")                            # sealed wire; fail-loud if unset

    rt = _runtime_cls(a.runtime)(a.model, LayerRange(a.layer_start, a.layer_end), a.device)

    lsock = _listen(a.bind_host, a.listen_port)  # listen BEFORE the (slow) load — peers connect into the backlog
    _load_serialized(rt, a.device)        # GPU: weights + flashinfer JIT (serialized per-GPU; mock: instant)
    if hasattr(rt, "warmup"):             # pre-compile flashinfer kernels NOW (at startup) so a promoted spare's
        t_w = time.time()                 # FIRST forward (the recovery) is instant, not a ~38s JIT compile (MTTR)
        rt.warmup()
        print(f"[serve] warmed flashinfer kernels in {time.time() - t_w:.1f}s", flush=True)
    print(f"[serve] loaded+warmed layers [{a.layer_start},{a.layer_end}) on {a.device}; "
          f"listening :{a.listen_port}, next {a.next_host}:{a.next_port}", flush=True)

    edge_out = LanEdge(a.next_host, a.next_port)
    _connect_retry(edge_out)              # dial next (it listens early too), then wait for our prev to dial in
    # Wait for our prev — but ALSO watch edge_out. If the DOWNSTREAM closes first, the run is being torn
    # down and no prev is ever coming: the canonical case is a pre-warmed standby SPARE in a no-death run
    # — never promoted, so a bare accept() blocks FOREVER and hangs teardown (the first live rec run +
    # `sky launch` both hung on exactly this). The spare's edge_out is connected to the driver; when the
    # driver exits it closes that socket → we see it readable (EOF) and exit cleanly. A prev dialing in
    # (we got promoted via the entry's set_next) wins the race and we serve normally.
    rlist, _, _ = select.select([lsock, edge_out], [], [])
    if lsock not in rlist:
        print("[serve] downstream closed before any prev connected — standby node exiting (never promoted)",
              flush=True)
        edge_out.close()
        lsock.close()
        return
    conn, _ = lsock.accept()
    edge_in = LanEdge.from_socket(conn)

    die_after = int(os.environ.get("CAIRN_DIE_AFTER", "0"))   # induced-death test hook: hard-exit after N forwards
    nfwd = 0
    while True:
        msg = edge_in.recv()
        if isinstance(msg, dict) and "op" in msg:
            if msg["op"] == "stop":
                try:
                    edge_out.send({"op": "stop"})
                except Exception:
                    pass
                break
            if msg["op"] == "set_next":           # live re-stitch: re-point edge_out at a new next (warm spare)
                try:
                    edge_out.close()
                except Exception:
                    pass
                edge_out = LanEdge(msg["host"], msg["port"])
                _connect_retry(edge_out)          # dial the spare (it's pre-warmed + listening)
                continue
            continue                              # unknown op — ignore
        h = msg["h"]
        if die_after and nfwd >= die_after:
            os._exit(137)                     # simulate a hard crash (spot reclaim) — this forward never returns
        if hasattr(h, "to"):
            h = h.to(a.device)
        out = rt.forward(h, {"seq": msg["seq"], "pos": msg["pos"]})
        if hasattr(out, "detach"):
            out = out.detach().cpu()
        edge_out.send({"h": out, "seq": msg["seq"], "pos": msg["pos"]})
        nfwd += 1

    edge_in.close()
    edge_out.close()
    lsock.close()


if __name__ == "__main__":
    main()
