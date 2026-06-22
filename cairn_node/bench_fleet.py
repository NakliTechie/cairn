"""cairn_node.bench_fleet — the v1.1 multi-stream bench driver (Chunk C). Run on rank0 of an already-running
fleet (the multi-stream counterpart of recovery_fleet). Over ONE persistent connection it runs three kinds
of decode round, then stops the fleet once:

  1. MULTI   — K streams concurrently (timed) → the occupancy signal (the pipeline stays full).
  2. SINGLE  — one stream alone (timed)       → the baseline for occupancy.
  3. CHECK   — each of the K streams SOLO (fresh seqs) → assert each MULTI stream == its SOLO run
               (cross-stream KV-isolation, the v1.1 HARD gate).

occupancy ≈ speedup / N, where speedup = (K-stream throughput) / (single-stream throughput); a full
pipeline gives speedup ≈ min(K, N) → occupancy ≈ 1. Emits a flushed verdict.

    SHARD_PSK=dev python -m cairn_node.bench_fleet                                   # CPU mock self-test
    python -m cairn_node.bench_fleet remote --head-port 7777 --sink-port 7779 --k 4 --n-new 16 --n-stages 4
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT / "fork") not in sys.path:
    sys.path.insert(0, str(_ROOT / "fork"))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from shard.transport import LanEdge  # noqa: E402
from shard import wire  # noqa: E402
from cairn_node.serve import _connect_retry, _listen  # noqa: E402
from cairn_node.pipeline import _drive_multi  # noqa: E402


def _streams(k: int):
    """K distinct prompts (distinct content + last token) so a cross-stream bleed would visibly corrupt
    a victim stream's output (vs its solo run)."""
    return [(f"k{i}", [i + 1, (i + 3) % 97 + 1, (i * 7 + 5) % 97 + 1, (i * 13 + 2) % 97 + 1]) for i in range(k)]


def run_bench(head, tail, k: int, n_new: int, n_stages: int, log=print) -> dict:
    """Run MULTI + SINGLE + CHECK rounds over an already-connected (head, tail); return a verdict dict.
    Distinct seqs per round so KV never collides on the shared nodes (no free needed for a small bench)."""
    streams = _streams(k)

    t0 = time.time()                                                    # 1. MULTI — K streams at once (timed)
    multi = _drive_multi(head, tail, [(f"m_{s}", p) for s, p in streams], n_new, stop=False)
    wall_multi = time.time() - t0

    t0 = time.time()                                                    # 2. SINGLE — one stream (the baseline)
    _drive_multi(head, tail, [("base", streams[0][1])], n_new, stop=False)
    wall_single = time.time() - t0

    mismatches = []                                                     # 3. CHECK — each stream SOLO vs its MULTI
    for s, p in streams:
        solo = _drive_multi(head, tail, [(f"c_{s}", p)], n_new, stop=False)
        if multi[f"m_{s}"] != solo[f"c_{s}"]:
            mismatches.append(s)

    try:
        head.send({"op": "stop"})
    except Exception:
        pass

    speedup = (k * wall_single / wall_multi) if wall_multi > 0 else 0.0
    occupancy = speedup / n_stages if n_stages else 0.0
    verdict = {
        "k": k, "n_new": n_new, "n_stages": n_stages,
        "wall_multi_s": round(wall_multi, 4), "wall_single_s": round(wall_single, 4),
        "speedup": round(speedup, 3), "occupancy": round(occupancy, 3),
        "isolation_ok": not mismatches, "mismatched_streams": mismatches,
    }
    log("[bench] K=%d n_new=%d N=%d | multi %.3fs single %.3fs | speedup %.2fx occupancy %.2f | isolation %s"
        % (k, n_new, n_stages, wall_multi, wall_single, speedup, occupancy,
           "OK" if not mismatches else "FAIL " + str(mismatches)))
    return verdict


def _remote(a) -> None:
    wire.key_from_env("SHARD_PSK")
    fh = open(a.log, "a", buffering=1) if a.log else None

    def log(msg):
        print(msg, flush=True)
        if fh is not None:
            fh.write(msg + "\n"); fh.flush(); os.fsync(fh.fileno())

    lsock = _listen(a.bind_host, a.sink_port)                          # tail dials back here
    head = LanEdge(a.head_host, a.head_port); _connect_retry(head)     # driver -> entry (local on rank0)
    conn, _ = lsock.accept(); tail = LanEdge.from_socket(conn)
    log(f"[bench] fleet connected; K={a.k} n_new={a.n_new} N={a.n_stages} — running gates")
    v = run_bench(head, tail, a.k, a.n_new, a.n_stages, log=log)
    log("[bench] VERDICT occupancy=%.3f (floor %.2f -> %s) | cross-stream isolation %s"
        % (v["occupancy"], a.floor, "PASS" if v["occupancy"] >= a.floor else "BELOW-FLOOR",
           "PASS" if v["isolation_ok"] else "FAIL"))
    if fh is not None:
        fh.close()


def _spawn_mock(ls, le, lp, np_):
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", "mock", "--model", "mock:8",
         "--layer-start", str(ls), "--layer-end", str(le), "--device", "cpu",
         "--listen-port", str(lp), "--bind-host", "127.0.0.1", "--next-host", "127.0.0.1", "--next-port", str(np_)],
        cwd=str(_ROOT), env=env)


def _selftest() -> None:
    """CPU mock: spawn a 2-stage fleet, run the bench gates, assert cross-stream isolation holds (the
    occupancy/timing numbers are meaningless on the instant, stateless mock — only the plumbing is tested)."""
    os.environ.setdefault("SHARD_PSK", "cairn-dev-psk")
    wire.key_from_env("SHARD_PSK")
    NL, CUT, base = 8, 4, 7860
    entry = _spawn_mock(0, CUT, base, base + 1)
    tail_p = _spawn_mock(CUT, NL, base + 1, base + 2)
    procs = [entry, tail_p]
    sink = _listen("127.0.0.1", base + 2)
    try:
        head = LanEdge("127.0.0.1", base); _connect_retry(head)
        conn, _ = sink.accept(); tail = LanEdge.from_socket(conn)
        v = run_bench(head, tail, k=4, n_new=6, n_stages=2)
        assert v["isolation_ok"], f"cross-stream isolation FAIL: {v['mismatched_streams']}"
        print(">>> BENCH HARNESS OK (multi-stream cross-stream isolation holds; timing meaningless on the mock)")
    finally:
        try:
            sink.close()
        except Exception:
            pass
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                p.kill()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("selftest", help="CPU mock bench self-test (default)")
    rp = sub.add_parser("remote", help="drive an already-running fleet (rank0)")
    rp.add_argument("--head-host", default="127.0.0.1")
    rp.add_argument("--head-port", type=int, required=True)
    rp.add_argument("--sink-port", type=int, required=True)
    rp.add_argument("--bind-host", default="0.0.0.0")
    rp.add_argument("--k", type=int, default=4)                        # concurrent streams
    rp.add_argument("--n-new", type=int, default=16)
    rp.add_argument("--n-stages", type=int, required=True)             # N (for occupancy = speedup / N)
    rp.add_argument("--floor", type=float, default=0.8)                # occupancy floor (spec §9.2)
    rp.add_argument("--log", default="/tmp/cairn-bench.log")
    a = ap.parse_args()
    if a.cmd == "remote":
        _remote(a)
    else:
        _selftest()


if __name__ == "__main__":
    main()
