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


def _is_draining(msg) -> bool:
    """The tail emitted a proactive-drain notice (it saw its own ~2-min spot warning) IN PLACE of this
    forward's result. A control dict, not a hidden-state result — so it must be caught before _next_tok."""
    return isinstance(msg, dict) and msg.get("op") == "draining"


def decode_with_recovery(head, active, spare, spare_host: str, spare_port: int,
                         prompt: List[int], n_new: int, seq: str = "s0", on_event=None):
    """Greedy-decode `n_new` tokens through head -> … -> active, surviving the loss of `active` onto the
    pre-warmed `spare`. TWO triggers, ONE recovery mechanism (set_next -> replay -> resume):

      • REACTIVE  — the node DIED: `active.recv()` raises (EDGE_ERRORS). There was a brief gap; the replay
                    rebuilds the spare's KV and re-derives the token we were mid-waiting-for.
      • PROACTIVE — the node is DRAINING: it saw its own ~2-min spot-interruption notice and, while still
                    alive, sent `{"op":"draining"}` in place of this forward's result. No error, no dropped
                    token — we pre-emptively re-stitch the entry to the spare + replay, so the STREAM
                    continues seamlessly. (The drained node then EXITS: the set_next below closes its
                    edge_in — "graceful" is about the stream, NOT the node surviving.) Reactive is the fallback.

    Both do the same thing: set_next (entry -> warm spare), replay the committed history under a FRESH seq
    (entry + fresh spare both prefill cleanly; the replay's last logit IS the pending token), resume on the
    spare. Returns (tokens, mttr_s | None).

    `on_event(kind, *payload)` — optional progress hook so a live run leaves a durable per-step trace
    ("tok", i, tok) / ("death", n_committed) / ("draining", n_committed) / ("recovered", mttr_s, tok, i).
    It must never raise; a logging failure cannot be allowed to break the decode (truthful-run discipline)."""
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

    def _recover_to_spare():
        """Re-stitch entry -> warm spare + replay the committed history; return (pending_tok, hist_len, mttr).
        Shared by both triggers — the difference is reactive had a gap (the node already died) while proactive
        did not (the switch is seamless: the node signalled while alive). Either way the set_next re-points the
        entry at the spare, which CLOSES the old node's edge_in — so even the drained (proactive) node then
        exits, not just the reactively-dead one. Reads the pending token FRESH from the spare."""
        nonlocal active, seq
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
        ptok = _next_tok(active.recv())                                        # RE-READ the pending token from the spare
        return ptok, len(hist), time.time() - t0

    while len(out) < n_new:
        head.send({"h": cur, "seq": seq, "pos": pos})
        try:
            msg = active.recv()
        except EDGE_ERRORS:
            # REACTIVE: the node died abruptly. Recover onto the spare (with the brief gap).
            if recovered:
                raise RuntimeError("second node death — out of warm spares (Path-1 single-spare scope)")
            _emit("death", len(out))                                                # committed tokens before death
            tok, hist_len, mttr = _recover_to_spare()
            out.append(tok)
            pos = hist_len
            cur = torch.tensor([[tok]])
            recovered = True
            _emit("recovered", mttr, tok, len(out))
            continue
        if _is_draining(msg):
            # PROACTIVE: the node is draining but STILL ALIVE — no error, no dropped token. Pre-emptively
            # re-stitch to the spare + replay (re-reads the pending token from the spare), then continue.
            if recovered:
                continue                                                            # already migrated; ignore a late notice
            _emit("draining", len(out))                                             # committed tokens at the drain notice
            tok, hist_len, mttr = _recover_to_spare()
            out.append(tok)
            pos = hist_len
            cur = torch.tensor([[tok]])
            recovered = True
            _emit("recovered", mttr, tok, len(out))
            continue
        tok = _next_tok(msg)
        out.append(tok)
        pos += cur.shape[1]
        cur = torch.tensor([[tok]])
        _emit("tok", len(out), tok)
    return out[:n_new], mttr


def decode_with_recovery_nstage(head, tail, spare, *, stage_addrs, spare_addr,
                                prompt: List[int], n_new: int, seq: str = "s0", on_event=None,
                                reshape_spare=None, partition=None):
    """N-stage generic-spare recovery: greedy-decode `n_new` tokens through entry(rank0) -> … -> tail,
    surviving the loss of ANY stage (entry / middle / tail) onto ONE generic warm `spare`.

      head        — edge → entry (rank 0); the driver sends here (re-pointed to the spare on ENTRY-replace).
      tail        — edge ← tail (rank N-1); the driver reads here (swapped to `spare` on TAIL-replace).
      spare       — edge ← the warm spare (spare → driver spare-sink); becomes the read edge iff the spare
                    replaces the TAIL (its launch edge_out already points at the driver's spare-sink).
      stage_addrs — [(host, listen_port)] per rank 0..N-1 (each stage's lsock) — used to fill the spare's
                    new downstream (k+1) on middle/entry replace.
      spare_addr  — (host, listen_port) of the spare's lsock (a predecessor / the driver dials it to promote).

    Trigger: PROACTIVE drain ({op:draining, stage:k} propagates downstream to the driver) covers ANY
    position. REACTIVE (the read edge raises) covers the TAIL (k=N-1); a reactive MIDDLE death needs
    per-stage heartbeats to identify k (Path 2) — out of scope here. ONE mechanism, three re-stitch cases:
      • tail   (k=N-1): set_next(target k-1 → spare); spare keeps edge_out→driver spare-sink → read `spare`.
      • middle (0<k<N-1): set_next(target k-1 → spare, spare_next=addr[k+1]); spare set_my_next→k+1; read tail.
      • entry  (k=0): driver re-points its OWN head → spare; spare set_my_next→addr[1]; read tail.
    Then REPLAY the committed history under a FRESH seq (every stage re-prefills → KV rebuilt; the replay's
    last logit IS the pending token), resume. The surviving SUCCESSOR of the dead stage re-accepts the spare
    on its lsock (serve.py rank>0 re-accept). Returns (tokens, mttr_s | None)."""
    import torch
    from shard.transport import LanEdge, EDGE_ERRORS
    N = len(stage_addrs)

    def _emit(kind, *p):
        if on_event is not None:
            try:
                on_event(kind, *p)
            except Exception:
                pass

    st = {"head": head, "read": tail, "seq": seq, "k": None, "spare": spare, "spare_addr": spare_addr}

    def _recover(k: int):
        """Re-stitch for a loss at rank k, replay the committed history, return (pending_tok, hist_len, mttr)."""
        t0 = time.time()
        st["k"] = k
        # LOAD-ON-PROMOTION: if a reshape callback is given, ask it for a spare shaped for rank k (it re-execs
        # an off-shape spare to load slice k from NVMe + returns its (edge, addr); same-shape returns instantly).
        if reshape_spare is not None:
            ls = le = None
            if partition is not None:
                ls = sum(partition[:k]); le = ls + int(partition[k])
            shaped = reshape_spare(k, ls, le)
            if shaped is not None:
                st["spare"], st["spare_addr"] = shaped
        if st["spare_addr"] is None or st["spare"] is None:
            raise RuntimeError(f"no usable spare for rank {k} recovery (reshape failed?)")
        sp_host, sp_port = st["spare_addr"]
        if k == 0:                                                  # ENTRY — the driver is the predecessor
            try:
                st["head"].close()
            except Exception:
                pass
            nh = LanEdge(sp_host, sp_port)
            _connect_retry(nh)                                      # driver → spare (promotes it: lsock accepts)
            nh.send({"op": "set_my_next", "host": stage_addrs[1][0], "port": stage_addrs[1][1]})
            st["head"] = nh                                         # send prompts to the spare now
        elif k == N - 1:                                            # TAIL — spare keeps edge_out→driver sink
            st["head"].send({"op": "set_next", "target_stage": k - 1, "host": sp_host, "port": sp_port})
            st["read"] = st["spare"]                                # results now arrive on the spare edge
        else:                                                       # MIDDLE — k-1 → spare → k+1
            st["head"].send({"op": "set_next", "target_stage": k - 1, "host": sp_host, "port": sp_port,
                             "spare_next_host": stage_addrs[k + 1][0], "spare_next_port": stage_addrs[k + 1][1]})
        st["seq"] = st["seq"] + "~r"                                # fresh seq → full re-prefill rebuilds KV
        hist = list(prompt) + out
        st["head"].send({"h": torch.tensor([hist]), "seq": st["seq"], "pos": 0})
        ptok = int(st["read"].recv()["h"][:, -1].argmax(-1))       # last logit = the pending token
        return ptok, len(hist), time.time() - t0

    cur = torch.tensor([list(prompt)])
    out: List[int] = []
    pos = 0
    mttr = None
    recovered = False
    while len(out) < n_new:
        st["head"].send({"h": cur, "seq": st["seq"], "pos": pos})
        try:
            msg = st["read"].recv()
        except EDGE_ERRORS:                                         # REACTIVE: read edge broke → tail death
            if recovered:
                raise RuntimeError("second node loss — out of warm spares (Path-1 single-spare scope)")
            _emit("death", N - 1, len(out))
            tok, hist_len, mttr = _recover(N - 1)
            out.append(tok); pos = hist_len; cur = torch.tensor([[tok]]); recovered = True
            _emit("recovered", mttr, tok, len(out)); continue
        if isinstance(msg, dict) and msg.get("op") == "draining":  # PROACTIVE: a stage signalled (carries k)
            if recovered:
                continue                                           # already migrated; ignore a late notice
            k = int(msg.get("stage", N - 1))
            _emit("draining", k, len(out))
            tok, hist_len, mttr = _recover(k)
            out.append(tok); pos = hist_len; cur = torch.tensor([[tok]]); recovered = True
            _emit("recovered", mttr, tok, len(out)); continue
        tok = int(msg["h"][:, -1].argmax(-1))
        out.append(tok); pos += cur.shape[1]; cur = torch.tensor([[tok]])
        _emit("tok", len(out), tok)
    # Do NOT send {op:stop} here — serve_http keeps the fleet up across requests. The caller stops it.
    # Returns the (possibly re-pointed) head/read edges + which rank was replaced + the promoted spare's addr,
    # so a PERSISTENT driver can update its topology: stage_addrs[k_dead] = that addr (it now serves rank k).
    return out[:n_new], mttr, st["head"], st["read"], st["k"], st["spare_addr"]


def decode_multi_with_recovery(head, tail, spare, *, stage_addrs, spare_addr, streams, n_new,
                               on_event=None, reshape_spare=None, partition=None):
    """K CONCURRENT streams + any-position warm-spare recovery — the multi-stream counterpart of
    decode_with_recovery_nstage merged with pipeline._drive_multi. Keeps one forward per stream in flight
    (so a full pipeline stays busy); on a node drop (drain{stage:k} or read-edge error) it re-stitches the
    dead stage onto the spare ONCE and replays EVERY live stream's committed history onto the healed
    pipeline (each replay's last logit = that stream's pending token), then resumes all streams.

    `streams` = [(seq, prompt), …]. Returns ({seq:[tokens]}, mttr_s|None, new_head, new_read, k_dead, spare_addr)
    — the trailing topology lets a persistent driver stay consistent across calls. Each stream's output is
    bit-identical to a no-drop run (the recovery is transparent per stream). Survives ONE drop (single spare)."""
    import torch
    from shard.transport import LanEdge, EDGE_ERRORS
    N = len(stage_addrs)

    def _emit(kind, *p):
        if on_event is not None:
            try:
                on_event(kind, *p)
            except Exception:
                pass

    prompt_of = {seq: list(p) for seq, p in streams}
    st = {seq: {"out": [], "pos": 0, "cur": torch.tensor([list(p)])} for seq, p in streams}
    gen = {seq: 0 for seq in st}                      # fresh-seq generation per stream (bumped on recovery)
    wire = {seq: seq for seq in st}                   # current on-wire seq per stream
    rev = {seq: seq for seq in st}                    # wire-seq -> base-seq (routing; stale seqs absent)
    sstate = {"head": head, "read": tail, "spare": spare, "spare_addr": spare_addr, "k": None}
    live = set(st)

    def _restitch(k):
        """Re-stitch the dead stage k onto the spare (reshape if off-shape). No replay (done per-stream)."""
        if reshape_spare is not None:
            ls = le = None
            if partition is not None:
                ls = sum(partition[:k]); le = ls + int(partition[k])
            shaped = reshape_spare(k, ls, le)
            if shaped is not None:
                sstate["spare"], sstate["spare_addr"] = shaped
        if sstate["spare_addr"] is None or sstate["spare"] is None:
            raise RuntimeError(f"no usable spare for rank {k} recovery")
        sp_host, sp_port = sstate["spare_addr"]
        if k == 0:                                                  # ENTRY
            try:
                sstate["head"].close()
            except Exception:
                pass
            nh = LanEdge(sp_host, sp_port); _connect_retry(nh)
            nh.send({"op": "set_my_next", "host": stage_addrs[1][0], "port": stage_addrs[1][1]})
            sstate["head"] = nh
        elif k == N - 1:                                            # TAIL
            sstate["head"].send({"op": "set_next", "target_stage": k - 1, "host": sp_host, "port": sp_port})
            sstate["read"] = sstate["spare"]
        else:                                                       # MIDDLE
            sstate["head"].send({"op": "set_next", "target_stage": k - 1, "host": sp_host, "port": sp_port,
                                 "spare_next_host": stage_addrs[k + 1][0], "spare_next_port": stage_addrs[k + 1][1]})

    def _recover(k):
        t0 = time.time()
        sstate["k"] = k
        _restitch(k)
        for base in list(live):                                    # fresh wire-seq per live stream
            rev.pop(wire[base], None)
            gen[base] += 1; wire[base] = f"{base}#r{gen[base]}"; rev[wire[base]] = base
        for base in list(live):                                    # replay each live stream's history
            hist = prompt_of[base] + st[base]["out"]
            sstate["head"].send({"h": torch.tensor([hist]), "seq": wire[base], "pos": 0})
        need = set(live)
        while need:                                                # collect replay results (skip stale/ops)
            m = sstate["read"].recv()
            if isinstance(m, dict) and m.get("op"):
                continue
            base = rev.get(m.get("seq"))
            if base is None or base not in need:
                continue
            hist_len = len(prompt_of[base]) + len(st[base]["out"])
            tok = int(m["h"][:, -1].argmax(-1))
            st[base]["out"].append(tok); st[base]["pos"] = hist_len; st[base]["cur"] = torch.tensor([[tok]])
            need.discard(base)
            if len(st[base]["out"]) >= n_new:
                live.discard(base)
        for base in live:                                          # re-inject the still-live streams
            sstate["head"].send({"h": st[base]["cur"], "seq": wire[base], "pos": st[base]["pos"]})
        return time.time() - t0

    for base in st:                                                # prime: one prefill per stream
        sstate["head"].send({"h": st[base]["cur"], "seq": wire[base], "pos": 0})
    mttr = None
    recovered = False
    while live:
        try:
            msg = sstate["read"].recv()
        except EDGE_ERRORS:
            if recovered:
                raise RuntimeError("second node loss — out of warm spares (Path-1 single-spare)")
            committed = sum(len(s["out"]) for s in st.values())
            _emit("death", N - 1, committed)
            mttr = _recover(N - 1); recovered = True
            _emit("recovered", mttr, -1, committed); continue
        if isinstance(msg, dict) and msg.get("op") == "draining":
            if recovered:
                continue
            k = int(msg.get("stage", N - 1))
            committed = sum(len(s["out"]) for s in st.values())
            _emit("draining", k, committed)
            mttr = _recover(k); recovered = True
            _emit("recovered", mttr, -1, committed); continue
        base = rev.get(msg.get("seq"))
        if base is None or base not in live:                       # stale (pre-recovery) result — ignore
            continue
        hist_len = len(prompt_of[base]) + len(st[base]["out"])
        st[base]["out"].append(int(msg["h"][:, -1].argmax(-1)))
        st[base]["pos"] = hist_len
        if len(st[base]["out"]) >= n_new:
            live.discard(base); continue
        st[base]["cur"] = torch.tensor([[st[base]["out"][-1]]])
        sstate["head"].send({"h": st[base]["cur"], "seq": wire[base], "pos": st[base]["pos"]})
    # also return the post-recovery topology so a persistent driver stays consistent across calls
    return ({seq: st[seq]["out"][:n_new] for seq in st}, mttr,
            sstate["head"], sstate["read"], sstate["k"], sstate["spare_addr"])


def _spawn_stage(runtime, model, ls, le, lp, nh, np_, rank, device="cpu",
                 drain_after=None, die_after=None, mem_fraction=None):
    """Spawn one cairn_node.serve with an explicit --stage-rank (N-stage), optionally inducing a
    proactive DRAIN (CAIRN_DRAIN_AFTER) or an abrupt DEATH (CAIRN_DIE_AFTER) after N forwards."""
    env = {**os.environ, "PYTHONPATH": str(_ROOT)}
    env.pop("CAIRN_DIE_AFTER", None)
    env.pop("CAIRN_DRAIN_AFTER", None)
    if drain_after:
        env["CAIRN_DRAIN_AFTER"] = str(drain_after)
    if die_after:
        env["CAIRN_DIE_AFTER"] = str(die_after)
    if mem_fraction:
        env["CAIRN_SGLANG_MEM_FRACTION"] = str(mem_fraction)
    return subprocess.Popen(
        [sys.executable, "-m", "cairn_node.serve", "--runtime", runtime, "--model", model,
         "--layer-start", str(ls), "--layer-end", str(le), "--device", device,
         "--listen-port", str(lp), "--bind-host", "127.0.0.1",
         "--next-host", nh, "--next-port", str(np_), "--stage-rank", str(rank)],
        cwd=str(_ROOT), env=env)


def prove_recovery_nstage(runtime: str, model: str, num_layers: int, cuts: List[int], prompt: List[int],
                          n_new: int, *, k_dead: int, drain_after: int, device: str = "cpu",
                          mem_fraction=None, base_port: int = 7600, log_path: str | None = None):
    """N-stage generic-spare recovery proof (all on localhost). Build the no-death reference, stand up N
    stages + 1 spare (shaped for position k_dead — generic load-on-promotion is a GPU/NVMe concern, not
    mockable), DRAIN stage k_dead after `drain_after` forwards, recover via decode_with_recovery_nstage,
    and assert recovered == reference. The CPU-mock proof of the re-stitch+replay WIRING for any position."""
    from shard.transport import LanEdge
    from shard import wire
    os.environ.setdefault("SHARD_PSK", "cairn-dev-psk")
    wire.key_from_env("SHARD_PSK")
    from cairn_node.pipeline import run_pipeline

    _fh = open(log_path, "a", buffering=1) if log_path else None

    def log(m: str) -> None:
        print(m, flush=True)
        if _fh is not None:
            _fh.write(m + "\n"); _fh.flush(); os.fsync(_fh.fileno())

    bnd = [0, *cuts, num_layers]
    N = len(bnd) - 1
    assert 0 <= k_dead < N, f"k_dead {k_dead} out of range for N={N}"
    log(f"[prove-N] runtime={runtime} N={N} cuts={cuts} k_dead={k_dead} drain_after={drain_after} "
        f"prompt={prompt} n_new={n_new}")

    ref = run_pipeline(runtime, model, num_layers, list(cuts), prompt, n_new, device=device, base_port=base_port)
    log(f"[prove-N] ref (no death) = {ref}")

    b = base_port + 20
    ports = [b + i for i in range(N)]
    spare_port, sink_t, sink_s = b + N, b + N + 1, b + N + 2
    procs = []
    for i in range(N):
        np_ = ports[i + 1] if i < N - 1 else sink_t
        procs.append(_spawn_stage(runtime, model, bnd[i], bnd[i + 1], ports[i], "127.0.0.1", np_, i,
                                  device=device, drain_after=(drain_after if i == k_dead else None),
                                  mem_fraction=mem_fraction))
    procs.append(_spawn_stage(runtime, model, bnd[k_dead], bnd[k_dead + 1], spare_port, "127.0.0.1", sink_s,
                              k_dead, device=device, mem_fraction=mem_fraction))   # generic spare (shaped for k_dead)

    st = _listen("127.0.0.1", sink_t)
    ss = _listen("127.0.0.1", sink_s)
    try:
        head = LanEdge("127.0.0.1", ports[0], supervised_recv_timeout=True); _connect_retry(head)
        ct, _ = st.accept(); tail_e = LanEdge.from_socket(ct, supervised_recv_timeout=True)
        cs, _ = ss.accept(); spare_e = LanEdge.from_socket(cs, supervised_recv_timeout=True)
        stage_addrs = [("127.0.0.1", p) for p in ports]

        def _on(kind, *p):
            if kind == "tok":
                log(f"[prove-N] tok {p[0]}/{n_new} = {p[1]}")
            elif kind == "draining":
                log(f"[prove-N] *** DRAIN at stage {p[0]} after {p[1]} committed — re-stitch + replay")
            elif kind == "death":
                log(f"[prove-N] *** DEATH at stage {p[0]} after {p[1]} committed — re-stitch + replay")
            elif kind == "recovered":
                log(f"[prove-N] *** RECOVERED in {p[0]:.3f}s — resumed at tok {p[2]} = {p[1]}")

        rec, mttr, new_head, _read, _k, _sa = decode_with_recovery_nstage(
            head, tail_e, spare_e, stage_addrs=stage_addrs, spare_addr=("127.0.0.1", spare_port),
            prompt=prompt, n_new=n_new, on_event=_on)
        try:
            new_head.send({"op": "stop"})                          # tear the (healed) pipeline down
        except Exception:
            pass
        ok = (rec == ref)
        log(f"[prove-N] rec (drain k={k_dead}) = {rec}  MATCH={ok}  "
            f"MTTR={('%.3fs' % mttr) if mttr is not None else 'none'}")
        if not ok:
            log(">>> N-STAGE RECOVERY FAIL — recovered != reference")
            raise AssertionError(f"recovered {rec} != reference {ref}")
        log(f">>> N-STAGE RECOVERY OK (k={k_dead}: drain → re-stitch → replay → resume == no-death)")
        return ref, rec, mttr
    finally:
        for s in (st, ss):
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
        # supervised read edges: a half-open peer (abrupt reclaim, no FIN) is caught as a death via the
        # generous CAIRN_EDGE_RECV_TIMEOUT deadline instead of blocking recv() forever (the live hang fix).
        head = LanEdge("127.0.0.1", pE, supervised_recv_timeout=True); _connect_retry(head)   # driver -> entry
        ct, _ = sink_t.accept(); tail_e = LanEdge.from_socket(ct, supervised_recv_timeout=True)   # tail  -> driver
        cs, _ = sink_s.accept(); spare_e = LanEdge.from_socket(cs, supervised_recv_timeout=True)  # spare -> driver (standby)

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
