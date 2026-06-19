"""The re-pointed LAN transport actually moves an activation tensor over a socket via
the sealed wire, and treats a tampered frame as a dead edge (spec §5.4/§6).

Needs torch + cryptography (the wire packs tensors + seals frames); skips cleanly when
they're absent so the default no-torch suite stays green. Run it with:
    uv run --python 3.9 --with torch --with numpy --with cryptography --with pytest pytest scheduler/tests/test_transport_lan.py
"""

import pathlib
import socket
import struct
import sys

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cryptography")

_FORK = pathlib.Path(__file__).resolve().parents[2] / "fork"
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

from shard import wire  # noqa: E402
from shard.transport import EDGE_ERRORS, LanEdge  # noqa: E402


def test_lan_edge_roundtrips_an_activation():
    wire.use_key("test-psk")
    a, b = socket.socketpair()
    tx = LanEdge.from_socket(a, name="s0->s1")
    rx = LanEdge.from_socket(b, name="rx")
    h = torch.randn(1, 4, 8, dtype=torch.bfloat16)

    tx.send({"op": "forward", "h": h, "pos": 3})
    got = rx.recv()

    assert got["op"] == "forward" and got["pos"] == 3
    assert torch.equal(got["h"], h)               # activation survived the wire exactly
    assert tx.bytes_sent > 0 and tx.health()["alive"]


def test_tampered_frame_is_a_dead_edge():
    wire.use_key("test-psk")
    a, b = socket.socketpair()
    rx = LanEdge.from_socket(b, name="rx")

    # a length-prefixed but unauthenticatable frame, straight onto the socket
    bad = b"\x00" * 40
    a.sendall(struct.pack("!Q", len(bad)) + bad)

    with pytest.raises(EDGE_ERRORS):
        rx.recv()
    assert rx.health()["alive"] is False and rx.resets == 1  # supervision marks it dead
