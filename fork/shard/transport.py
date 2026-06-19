"""inter-stage transport — re-pointed from Shard's WAN design to Cairn's VPC LAN.

Shard moved the activation tensor across the public internet between home GPUs: QUIC +
NAT hole-punching + relay fallback + a quantized activation codec. Cairn owns every node
in one VPC / placement group (invariant #6), so all of that is **deleted**: there is no
NAT to punch, RTT is sub-millisecond, and the activation tensor (a single KB-scale
hidden state per token) is far under LAN bandwidth (spec §6). What remains — and is
**load-bearing for recovery detection, do not weaken** (handoff §3) — is the *supervised
edge*: every edge logs its own health so a dead stage is detected fast, and the wire is
the sealed, pickle-free framing kept verbatim from the fork.

`wire` is imported lazily (it pulls in torch for tensor packing) so this module imports
without a GPU/torch toolchain for structural checks.
"""

from __future__ import annotations

import socket
import time
from typing import Any, Optional, Tuple

# A frame we can't authenticate+parse is surfaced by wire as ConnectionError; these are
# the errors the per-edge supervision treats as "dead edge → trigger recovery" (spec §5.4).
EDGE_ERRORS: Tuple[type, ...] = (ConnectionError, OSError, socket.timeout)


class ActivationCodec:
    """Off by default in-VPC (handoff §11.9): at LAN bandwidth, quantising the activation
    tensor buys nothing, so v1.0 ships identity passthrough. The fp8/int8 quant is kept as
    a knob for the v1.2 AZ-spread case where hops cross the network."""

    def __init__(self, mode: str = "off") -> None:
        if mode not in ("off", "fp8", "int8"):
            raise ValueError(f"unknown codec mode {mode!r}")
        self.mode = mode

    def encode(self, hidden_states: Any) -> Any:
        if self.mode == "off":
            return hidden_states
        raise NotImplementedError("activation quant codec is a v1.2 AZ-spread knob")

    def decode(self, payload: Any) -> Any:
        if self.mode == "off":
            return payload
        raise NotImplementedError("activation quant codec is a v1.2 AZ-spread knob")


class LanEdge:
    """One supervised pipeline edge: a plain TCP connection from this stage to the next,
    inside the VPC. No rendezvous, hole-punch, or relay — direct dial on the LAN. Carries
    activations as sealed wire frames; surfaces health() so a stalled edge is caught fast."""

    def __init__(self, peer_host: str, peer_port: int, *, name: str = "",
                 connect_timeout: float = 5.0, codec: Optional[ActivationCodec] = None) -> None:
        self.peer_host = peer_host
        self.peer_port = peer_port
        self.name = name or f"{peer_host}:{peer_port}"
        self.connect_timeout = connect_timeout
        self.codec = codec or ActivationCodec("off")
        self._sock: Optional[socket.socket] = None
        # health — the thing a black-box binary never gave us (spec §10: per-edge health)
        self.last_ok: float = 0.0
        self.bytes_sent: int = 0
        self.resets: int = 0
        self.alive: bool = False

    @classmethod
    def from_socket(cls, sock: socket.socket, **kw: Any) -> "LanEdge":
        """Wrap an already-accepted socket (the receiving side of an edge)."""
        edge = cls("", 0, **kw)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        edge._sock = sock
        edge.alive = True
        return edge

    def connect(self) -> None:
        """Direct LAN dial (no hole-punch). Raises on failure → supervision retries."""
        sock = socket.create_connection((self.peer_host, self.peer_port), timeout=self.connect_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._sock = sock
        self.alive = True
        self.last_ok = time.monotonic()

    def send(self, obj: Any) -> None:
        if self._sock is None:
            raise ConnectionError(f"edge {self.name} not connected")
        from . import wire  # lazy: wire pulls in torch
        try:
            n = wire.send_msg(self._sock, self.codec.encode(obj))
            self.bytes_sent += n
            self.last_ok = time.monotonic()
        except EDGE_ERRORS:
            self._die()
            raise

    def recv(self) -> Any:
        if self._sock is None:
            raise ConnectionError(f"edge {self.name} not connected")
        from . import wire  # lazy
        try:
            obj = self.codec.decode(wire.recv_msg(self._sock))
            self.last_ok = time.monotonic()
            return obj
        except EDGE_ERRORS:
            self._die()
            raise

    def _die(self) -> None:
        self.alive = False
        self.resets += 1
        self.close()

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            finally:
                self._sock = None

    def health(self) -> dict:
        """rtt-free liveness snapshot the scheduler/recovery loop reads (spec §5.4/§10)."""
        return {
            "name": self.name,
            "alive": self.alive,
            "age_s": (time.monotonic() - self.last_ok) if self.last_ok else None,
            "bytes_sent": self.bytes_sent,
            "resets": self.resets,
        }
