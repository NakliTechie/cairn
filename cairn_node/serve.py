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
    # N-stage recovery (2026-06-23): a stage knows its own rank so a proactive-drain notice can
    # tell the driver WHICH stage is draining (the driver then re-stitches the dead stage's
    # PREDECESSOR, not always the entry). Default 0 = backwards-compatible with the 2-stage
    # entry+tail run (the driver only cares about the tail's edge_in, and its target_stage
    # defaults to 0 so set_next reaches the entry just like before).
    ap.add_argument("--stage-rank", type=int, default=0,
                    help="position in the pipeline (0=entry; used by N-stage recovery to address set_next)")
    ap.add_argument("--announce", action="store_true",
                    help="standby spare: send a hello{host,port} on edge_out so the driver learns this "
                         "node's listen addr for promotion (needed for replenished spares with new IPs)")
    ap.add_argument("--advertise-host", default="",
                    help="host to advertise in the --announce hello (the box's routable IP; default bind-host)")
    a = ap.parse_args()

    from shard.node import LayerRange
    from shard.transport import LanEdge, EDGE_ERRORS
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
    # A standby SPARE announces its own listen addr to the driver (its edge_out → the driver's spare-sink)
    # so the driver learns where to dial to PROMOTE it — essential for REPLENISHED spares (a freshly
    # provisioned box has a new IP the driver didn't know at launch). One-shot; the driver reads it as the
    # spare's hello, then the spare waits idle until promoted. Off the data path (no effect once serving).
    if a.announce:
        adv = a.advertise_host or a.bind_host
        try:
            edge_out.send({"op": "hello", "host": adv, "port": a.listen_port})
            print(f"[serve] announced standby addr {adv}:{a.listen_port} to the driver", flush=True)
        except Exception:
            pass
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

    # PROACTIVE drain — the spot-interruption (~2-min) notice fires WHILE this node is still alive, so the
    # driver can migrate the in-flight STREAM to the warm spare with no dropped token (graceful) instead of
    # waiting for a death. `_draining` is a 1-slot flag flipped by either the spot_watch thread (gated on
    # CAIRN_SPOT_WATCH so tests/CI never poll real IMDS) OR a test op `{"op":"drain"}` injected into the
    # pipeline. When set, the wire loop emits ONE `{"op":"draining"}` to edge_out (→ the driver) in place of
    # a forward result, then loops back to recv.
    #   NOTE — this node does NOT stay alive after the drain. Once the driver re-stitches (it sends the ENTRY
    #   a `set_next` pointing at the spare — recovery.py), the entry CLOSES this node's edge_in, so our very
    #   next recv() sees peer-closed and this process EXITS (serve.py:161, ~150ms later — measured CPU repro
    #   2026-06-23). "Graceful" is about the STREAM (it migrates seamlessly, no gap); the drained node still
    #   dies — which is fine, its box is being reclaimed anyway. See recovery.py + test_proactive_drain.py.
    # A list (not a bare bool) so the watch thread's closure mutates the SAME object the loop reads.
    _draining = [False]

    def _on_spot_notice(notice):
        print(f"[serve] *** spot-interruption notice — DRAINING (still alive): {notice}", flush=True)
        _draining[0] = True
        try:                                                  # signal the HUMAN — the earliest warning we get
            import socket as _sock
            from cairn_node import notify as _notify
            _notify.alert(f"spot reclaim incoming on {_sock.gethostname()} "
                          f"(pp_rank={os.environ.get('CAIRN_PP_RANK', '?')}) — node draining onto the spare; "
                          f"the run may end. notice={notice.get('action', notice)}")
        except Exception:
            pass

    # Real path: a background thread polls THIS box's IMDS for its own ~2-min spot notice (opt-in, so
    # tests/CI never poll the live 169.254.169.254). Test path: CAIRN_SPOT_TEST_FILE points at a sentinel
    # the test `touch`es mid-decode — a local stand-in for IMDS that drives the SAME drain wire path with
    # no EC2. Either way the watcher only flips `_draining`; the wire loop does the one-shot emit. Both run
    # off the hot path (a daemon thread), so steady-state serving pays nothing.
    _spot_file = os.environ.get("CAIRN_SPOT_TEST_FILE")
    if os.environ.get("CAIRN_SPOT_WATCH") == "1" or _spot_file:
        import threading
        interval = float(os.environ.get("CAIRN_SPOT_WATCH_INTERVAL", "5"))
        if _spot_file:
            poll = float(os.environ.get("CAIRN_SPOT_WATCH_INTERVAL", "0.005"))
            def _watch_file():
                while not _draining[0]:
                    if os.path.exists(_spot_file):
                        _on_spot_notice({"_kind": "test-file", "action": "terminate", "path": _spot_file})
                        return
                    time.sleep(poll)
            threading.Thread(target=_watch_file, daemon=True).start()
            print(f"[serve] spot drain armed via sentinel file {_spot_file} (test stand-in for IMDS)", flush=True)
        else:
            from cairn_node.spot_watch import watch as _spot_watch
            threading.Thread(target=lambda: _spot_watch(_on_spot_notice, interval=interval),
                             daemon=True).start()
            print(f"[serve] spot_watch armed (interval={interval}s) — will drain on the ~2-min notice", flush=True)

    die_after = int(os.environ.get("CAIRN_DIE_AFTER", "0"))   # induced-death test hook: hard-exit after N forwards
    drain_after = int(os.environ.get("CAIRN_DRAIN_AFTER", "0"))  # induced-DRAIN hook (proactive): drain after N forwards
    drained_sent = False                                      # emit the draining op AT MOST once
    nfwd = 0
    while True:
        try:
            msg = edge_in.recv()
        except EDGE_ERRORS:
            # The ONE expected recv failure here is the post-drain teardown. After we emitted the draining
            # notice, the driver re-stitches the ENTRY to the warm spare (recovery.py `_recover_to_spare`'s
            # set_next), which CLOSES our edge_in — so this recv hits peer-closed. That is the EXPECTED end
            # of a drained tail's life (its box is being reclaimed anyway): log it and exit CLEANLY (rc=0)
            # instead of dying on an unhandled ConnectionError (rc=1, which the serve yaml's `exit 0` only
            # masked). CRITICAL: only when we ACTUALLY drained. If drained_sent is False this is a REAL
            # upstream death — re-raise so it keeps its current fail-loud behaviour (a genuine peer death
            # must NOT be swallowed). See test_proactive_drain.py + plan/workplan.md Chunk 0.
            if drained_sent:
                print("[serve] drained tail: entry re-stitched to the spare, edge_in closed — "
                      "exiting cleanly", flush=True)
                break
            # N-STAGE RECOVERY (2026-06-28): a SURVIVING stage whose PREDECESSOR was just replaced (i.e. the
            # dead/drained node's successor) sees its edge_in break. Instead of dying, RE-ACCEPT a new
            # predecessor on lsock — the promoted spare (or a re-stitched k-1) dials in. Bounded by
            # CAIRN_REACCEPT_TIMEOUT; a timeout = no replacement is coming (genuine catastrophic failure,
            # Path 2's job) → re-raise fail-loud. rank 0 (entry) NEVER re-accepts: its predecessor is the
            # driver, which re-points its own outbound to the spare directly (recovery.py entry-replace).
            if a.stage_rank > 0:
                ra = float(os.environ.get("CAIRN_REACCEPT_TIMEOUT", "150"))
                print(f"[serve] rank {a.stage_rank}: edge_in closed (predecessor replaced?) — re-accepting "
                      f"a new predecessor on :{a.listen_port} (timeout {ra}s)", flush=True)
                lsock.settimeout(ra)
                try:
                    conn2, _ = lsock.accept()
                except (OSError, socket.timeout):
                    print(f"[serve] rank {a.stage_rank}: re-accept timed out — no replacement; re-raising",
                          flush=True)
                    raise
                lsock.settimeout(None)
                try:
                    edge_in.close()
                except Exception:
                    pass
                edge_in = LanEdge.from_socket(conn2)
                print(f"[serve] rank {a.stage_rank}: re-accepted new predecessor — resuming", flush=True)
                continue
            raise
        if isinstance(msg, dict) and "op" in msg:
            if msg["op"] == "stop":
                try:
                    edge_out.send({"op": "stop"})
                except Exception:
                    pass
                break
            if msg["op"] == "set_next":           # live re-stitch: re-point edge_out at a new next (warm spare)
                # N-stage routing (2026-06-23): if target_stage is set and ISN'T me, forward the
                # message down the wire so it reaches the stage that actually owns the re-stitch.
                # 2-stage runs (and any set_next without target_stage) keep the original behavior:
                # the first stage receiving it consumes — which for the 2-stage entry+tail case
                # is the entry, and that's correct (it's the tail's predecessor).
                target = msg.get("target_stage")
                if target is not None and target != a.stage_rank:
                    try:
                        edge_out.send(msg)
                    except Exception:
                        pass
                    continue
                try:
                    edge_out.close()
                except Exception:
                    pass
                edge_out = LanEdge(msg["host"], msg["port"])
                _connect_retry(edge_out)          # dial the spare (it's pre-warmed + listening)
                # If the driver bundled the spare's downstream address in this same set_next, the
                # spare is replacing a stage that ISN'T the tail — its launch-configured edge_out
                # (driver's sink) is the wrong target. Tell the spare its new downstream now,
                # BEFORE any data forwards: spare's edge_in just connected to us, so it'll recv
                # this set_my_next as its first message and rewire before reading anything else.
                if "spare_next_host" in msg and "spare_next_port" in msg:
                    try:
                        edge_out.send({"op": "set_my_next",
                                       "host": msg["spare_next_host"],
                                       "port": msg["spare_next_port"]})
                    except Exception:
                        pass
                continue
            if msg["op"] == "set_my_next":        # spare promotion: re-point MY edge_out (received as first msg)
                try:
                    edge_out.close()
                except Exception:
                    pass
                edge_out = LanEdge(msg["host"], msg["port"])
                _connect_retry(edge_out)
                continue
            if msg["op"] == "drain":              # test hook: inject a drain WITHOUT real IMDS (== a spot notice)
                _draining[0] = True
                continue
            if msg["op"] == "draining":           # forwarded from an UPSTREAM stage — propagate so driver hears it
                try:
                    edge_out.send(msg)
                except Exception:
                    pass
                continue
            continue                              # unknown op — ignore
        if drain_after and nfwd >= drain_after:   # induced-drain test hook (deterministic proactive drain)
            _draining[0] = True
        # Draining? Tell the driver ONCE (it pre-emptively re-stitches the entry to the warm spare), then
        # loop back to recv. We send the notice in PLACE of this forward's result: the driver reads it as the
        # response to its in-flight send, re-stitches + replays through the spare for the pending token, and
        # abandons our (orphaned) reply. We do NOT keep serving — the re-stitch closes our edge_in, so the
        # next recv() raises peer-closed and we exit (the box is being reclaimed anyway). Off the hot path.
        if _draining[0] and not drained_sent:
            try:
                # N-stage: include OUR rank so the driver knows WHICH stage is draining and can
                # send set_next to the right target (= our rank - 1). 2-stage default rank=0 means
                # the field is present but the driver's k_dead - 1 = -1 case is the entry-replace
                # path (driver re-stitches its own outbound) — see cairn_node/recovery.py.
                edge_out.send({"op": "draining", "stage": a.stage_rank})
                drained_sent = True
                nfwd += 1
                continue                          # this forward is redone by the driver's replay on the spare
            except EDGE_ERRORS:
                drained_sent = True               # downstream already gone — fall through to the reactive path
        h = msg["h"]
        if die_after and nfwd >= die_after:
            os._exit(137)                     # simulate a hard crash (spot reclaim) — this forward never returns
        if hasattr(h, "to"):
            h = h.to(a.device)
        out = rt.forward(h, {"seq": msg["seq"], "pos": msg["pos"]})
        if hasattr(out, "detach"):
            out = out.detach().cpu()
        try:
            edge_out.send({"h": out, "seq": msg["seq"], "pos": msg["pos"]})
        except EDGE_ERRORS:
            # our NEXT node died ABRUPTLY (a real spot reclaim — the case a graceful CAIRN_DIE_AFTER exit
            # hides, because there the next recv'd our send before exiting). DON'T crash: the driver detects
            # the death (its own recv from the tail-sink fails), re-stitches us via `set_next`, and the
            # recovery REPLAY redoes this forward. Discard it and loop back to recv — the next message is
            # that set_next. (Without this the entry crashes here and recovery can never re-stitch it.)
            pass
        nfwd += 1

    edge_in.close()
    edge_out.close()
    lsock.close()


if __name__ == "__main__":
    main()
