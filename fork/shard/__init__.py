"""shard (Cairn hard fork) — the per-node block runtime + wire + LAN transport.

Forked from leyten/shard @ 73a6d53 and stripped to a LAN-in-one-VPC deployment
(see ../UPSTREAM.md). What remains:

  - node.py       NodeRuntime — serve one contiguous block of layers (SGLang wrap).
  - wire.py       the sealed, pickle-free framing (ChaCha20-Poly1305 / SHARD_PSK). KEPT verbatim.
  - transport.py  supervised pipeline edges, re-pointed from WAN/QUIC to VPC LAN TCP.
  - topology.py   min-latency loop solver (pure) — used only for v1.2 AZ-spread; in one
                  placement group the order is trivial.

Stripped (Cairn owns every node in one VPC — invariant #6): NAT/hole-punch/relay,
the activation codec hot path, c0mpute/payments/referrals, the permissionless
proof/receipt + per-node identity, spec-decode (v1.3+), and the research swarm
drivers. The scheduler/control plane is Cairn's own (`cairn_scheduler`, the net-new),
not this fork.
"""

__version__ = "0.1.0-cairn"
__upstream__ = "leyten/shard@73a6d533686b99e1207b0228d2ab26828623e675"
