# fork/ — stripped Shard fork (block runtime + LAN transport + wire)

Hard fork of [`leyten/shard`](https://github.com/leyten/shard), pinned to
`73a6d53` and stripped to a LAN-in-one-VPC deployment. Full keep/strip/re-point ledger:
[`UPSTREAM.md`](UPSTREAM.md).

```
fork/
  shard/
    wire.py        the sealed, pickle-free wire (ChaCha20-Poly1305 / SHARD_PSK). KEPT verbatim.
    node.py        NodeRuntime — serve one contiguous block of layers (SGLang wrap).
    transport.py   supervised LAN edge over the wire (WAN/QUIC/hole-punch/relay stripped).
    topology.py    pure min-latency loop solver — v1.2 AZ-spread only (trivial in one PG).
  adapter.py       ShardBlockRuntime → cairn_scheduler.runtime.BlockRuntime seam.
  LICENSE          Apache-2.0 © leyten (kept).
  UPSTREAM.md      provenance + keep/strip/re-point ledger.
```

## Status
- **Wire:** kept verbatim; crypto self-test passes on CPU (the v1.0 crypto gate, handoff §8) — run `uv run --with cryptography --with numpy --with torch python fork/shard/wire.py`.
- **Adapter:** conforms to the `BlockRuntime` seam (verified on CPU in `scheduler/tests/test_shard_adapter.py`); the SGLang `forward`/`load` are **GPU-gated** (rung 2/3) and raise `NotImplementedError` until built on a real g6 pool.
- **Transport:** re-pointed to LAN TCP+wire; the supervised health/reset path (load-bearing for recovery detection) is kept.

## Plugging into the scheduler
The same `cairn_scheduler.Scheduler` drives sim or real:
```python
from cairn_scheduler import load_model_config, fit
from cairn_scheduler.runtime import build_mock_pipeline      # rung 1 (no GPU)
# from adapter import build_shard_pipeline                    # rung 2/3 (GPU)
cfg = load_model_config("configs/llama-3.1-8b.yaml")
pipeline = build_mock_pipeline(fit(cfg))                      # ← swap to build_shard_pipeline on a GPU pool
```

**Not started:** the SGLang block-serving impl behind `NodeRuntime.forward` (rung 2 — needs a GPU + HF token for weights).
