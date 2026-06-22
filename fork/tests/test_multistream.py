"""Multi-stream harness (Chunk C v1.1) — K concurrent streams, each stream's output == its solo run.

This is the **cross-stream KV-isolation** hard gate: running K streams at once must not corrupt any one
of them. On the CPU mock it proves the DRIVER plumbing — per-`seq` routing, no cross-stream bleed (if the
driver mis-routed one stream's logit to another, the victim's next input would be wrong and its output
would diverge from its solo run). The real paged-KV isolation is the sglang case, GPU-gated (run on the
box): there the nodes keep a separate KV pool per seq, and the same test asserts no collision.

The same harness (`run_pipeline(..., streams=...)`) also backs the **occupancy** gate — timing K streams
vs a single stream gives the pipeline-fill speedup — but that needs real GPU timing, so it's measured on
the box, not here.

torch/cryptography-gated."""
import os
import pathlib
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
for _p in (str(_ROOT), str(_ROOT / "fork")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("torch")
pytest.importorskip("cryptography")
os.environ.setdefault("SHARD_PSK", "cairn-test-psk")

from cairn_node.pipeline import run_pipeline  # noqa: E402

NEW = 6
# distinct prompts, distinct LAST tokens (the mock's next token = last_id % H), varying lengths — so a
# mis-routed stream would visibly diverge from its solo run.
STREAMS = [("a", [1, 5, 3]), ("b", [2, 6, 7, 4]), ("c", [9, 8, 1, 5, 2])]


@pytest.mark.parametrize("cuts", [[4], [3, 6]], ids=["2-stage", "3-stage"])
def test_mock_multistream_isolation(cuts):
    """K=3 concurrent streams over a node-per-process pipeline; each stream's multi-stream output ==
    its solo (single-stream) run — no cross-stream bleed (CPU mock proves the driver's per-seq routing)."""
    base = 7700 + 80 * len(cuts)
    solo = {seq: run_pipeline("mock", "mock:8", 8, cuts, prompt, NEW, base_port=base + 15 * i)
            for i, (seq, prompt) in enumerate(STREAMS)}
    multi = run_pipeline("mock", "mock:8", 8, cuts, [], NEW, base_port=base + 900, streams=STREAMS)

    assert set(multi) == set(solo)
    for seq in solo:
        assert multi[seq] == solo[seq], f"stream {seq}: multi {multi[seq]} != solo {solo[seq]} (cross-stream bleed)"
    # sanity: the streams genuinely differ (so the isolation assertion above is meaningful, not trivially equal)
    assert len({tuple(v) for v in solo.values()}) == len(STREAMS)
