# Fork provenance — leyten/shard

**Upstream:** https://github.com/leyten/shard
**Pinned commit:** `73a6d533686b99e1207b0228d2ab26828623e675` (branch `master`, 2026)
**License:** Apache-2.0 © 2026 leyten (kept verbatim in [`LICENSE`](LICENSE) — handoff §11.11)

This is a **hard fork** (Build doctrine): pinned to the commit above, stripped, and
**not kept mergeable** with upstream. Shard targets a *permissionless WAN swarm of
consumer GPUs over the public internet*; Cairn owns every node in one VPC/placement
group (invariant #6), which deletes Shard's three hardest problems (NAT transport,
volunteer-node privacy, decentralised payments) — see `docs/cairn-vision-roadmap.md`.

## Kept

| Cairn path | Upstream | Why |
|---|---|---|
| `shard/wire.py` | `phase0/wire.py` | The wire — pickle-free JSON header + raw tensor bytes, ChaCha20-Poly1305 under `SHARD_PSK`, crypto self-test at boot (handoff §3/§6, spec §8). Verbatim **except one Cairn patch**: a `_MAX_FRAME` cap in `recv_msg` (forward-pass M11) — the pre-auth `!Q` length is now bounded so a hostile frame can't exhaust memory. |
| `shard/node.py` | `shard/node.py` | `NodeRuntime` — the `forward(hidden_states, kv_meta) → hidden` block-runtime contract (SGLang wrap). The seam the scheduler drives. |
| `shard/topology.py` | `shard/topology.py` | Pure min-latency loop solver (Held-Karp + 2-opt). **Not load-bearing in v1.0** (one placement group → trivial order); kept for v1.2 AZ-spread. |
| `LICENSE` | `LICENSE` | Apache-2.0, required. |

## Re-pointed (WAN → LAN)

| Cairn path | Change |
|---|---|
| `shard/transport.py` | Rewritten. Shard's `Edge` was QUIC + NAT hole-punch + relay fallback + a quantized activation codec. Cairn: a **supervised TCP edge over the wire**, direct LAN dial, no rendezvous/relay; the activation codec is an **off-by-default knob** (handoff §11.9). The supervised health/timeout/reset path is kept — it is how a dead stage is detected (handoff §3, "do not weaken"). |
| `adapter.py` | New — `ShardBlockRuntime` adapts `NodeRuntime` to `cairn_scheduler.runtime.BlockRuntime`, so the scheduler runs sim (`build_mock_pipeline`) or real GPU (`build_shard_pipeline`) unchanged. |

## Stripped (deleted — not vendored)

| Upstream | Reason |
|---|---|
| `phase0/proof_receipt.py` | Permissionless-swarm proof/identity (distinct public IPs / GPU UUIDs / multiple regions / WAN-scale edges) — the exact opposite of Cairn's owned single-VPC LAN. PSK suffices (spec §8). |
| `phase0/mesh.py` | WAN RTT measurement harness — not needed in one placement group (v1.0); a v1.2 AZ-spread tool. |
| `phase0/specdec.py`, `specpipe.py`, `specbench.py`, `fastverify.py`, `tree.py` + research spec drivers | Speculative decoding — **v1.3+ deferred** (handoff §11.8). |
| `shard/scheduler.py` | Stub control plane referencing the c0mpute orchestrator — **replaced by Cairn's own `cairn_scheduler`** (the net-new build). |
| `phase0/node*.py`, `pipeline.py`, `launch_*.py`, `bench.py`, `get_model.py`, `setup_box.sh` | Shard's WAN phase-0 implementation — re-implemented for LAN against the kept `NodeRuntime` contract when the SGLang forward lands (rung 2, GPU). |
| `research/` (entire dir, ~45 files) | Swarm-driver experiments, quant probes, CUDA-graph trials — research scaffolding, not the product. |
| `docs/` (ARCHITECTURE/PROOF/ROADMAP + research) | Upstream's WAN/permissionless narrative; Cairn's design lives in `../docs/`. |

## Re-sync policy

None. Pinned and stripped; upstream improvements are cherry-picked deliberately, never merged.
