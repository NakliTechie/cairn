"""Multi-process pipeline (cairn_node) — split == single-ref ACROSS PROCESSES, over the real sealed wire.

`test_sglang_split.py` proves the block-forward on ONE GPU in-process; this proves the cairn_node SERVING
layer composes across SEPARATE node processes — which the in-process harness can't, because two sglang
ModelRunners can't co-exist in one process (the global tensor-parallel group). Each node is its own
process (its own ModelRunner) wired to the next by fork/shard/transport.LanEdge over the sealed wire.

  - mock case: CPU, runs anywhere with torch — validates the serve loop + wire + driver PLUMBING.
  - sglang case: GPU + sglang gated (run on the box). Set CAIRN_SGLANG_MEM_FRACTION low when both nodes
    share one GPU (the dev-on-1-L4 case); per-GPU devices is a trivial follow-on for the true-2-GPU run.

torch/cryptography-gated (skipped without them)."""
import importlib.util
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

import torch  # noqa: E402

from cairn_node.pipeline import run_pipeline  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]
NEW = 6


@pytest.mark.parametrize("cuts", [[4], [3, 6]], ids=["2-proc", "3-proc"])
def test_mock_split_equals_unsplit_over_wire(cuts):
    """Plumbing: node-per-process wired over localhost; split == unsplit token-for-token (CPU mock)."""
    base = 7600 + 200 * len(cuts)
    ref = run_pipeline("mock", "mock:8", 8, [], PROMPT, NEW, base_port=base)
    assert run_pipeline("mock", "mock:8", 8, cuts, PROMPT, NEW, base_port=base + 100) == ref


@pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("sglang")),
    reason="GPU + sglang only (run on the box)",
)
def test_sglang_split_equals_unsplit_over_wire():
    """The real gate: 2 sglang node PROCESSES wired over localhost, split == single-ref token-for-token."""
    model = os.environ.get("CAIRN_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(model).num_hidden_layers
    ref = run_pipeline("sglang", model, n, [], PROMPT, NEW, device="cuda:0", base_port=7900)
    split = run_pipeline("sglang", model, n, [n // 2], PROMPT, NEW, device="cuda:0", base_port=7950)
    assert split == ref


@pytest.mark.skipif(
    not (torch.cuda.is_available() and importlib.util.find_spec("sglang")),
    reason="GPU + sglang only (run on the box)",
)
def test_sglang_replay_rebuild_over_wire():
    """Recovery = replay, across processes: a FRESH process-pipeline given only the token history
    rebuilds the paged KV (each node re-prefills) and resumes bit-identically — the over-the-wire
    counterpart of test_sglang_split's replay-rebuild (inv #2)."""
    model = os.environ.get("CAIRN_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    from transformers import AutoConfig
    n = AutoConfig.from_pretrained(model).num_hidden_layers
    kill_at = 3
    full = run_pipeline("sglang", model, n, [n // 2], PROMPT, NEW, device="cuda:0", base_port=8000)
    history = PROMPT + full[:kill_at]
    resumed = run_pipeline("sglang", model, n, [n // 2], history, NEW - kill_at, device="cuda:0", base_port=8050)
    assert resumed == full[kill_at:]
