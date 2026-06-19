# fork/ — stripped Shard fork (block runtime + LAN transport)

Hard fork of [`leyten/shard`](https://github.com/leyten/shard), pinned to a specific
upstream commit (handoff §3). **Keep + re-point:** contiguous-layer split + the block
runtime (SGLang wrap, `forward(hidden, kv_meta) → hidden`), supervised edges
(health/timeout/reconnect) re-pointed at the VPC LAN, the wire format (JSON header +
raw tensor bytes, **no pickle**, ChaCha20-Poly1305 under `SHARD_PSK`, crypto self-test
at boot). **Strip:** NAT/STUN/relay, c0mpute/payments/referrals/permissionless-join,
the privacy/boundary-pinning code, per-node identity issuance.

The real implementation of the `cairn_scheduler.runtime.BlockRuntime` seam plugs in
here at cost-ladder rungs 2–3 (handoff §6). **Status: not started** — deferred until
the rung-1 mock seam is proven. Needs a GPU to be meaningful.
