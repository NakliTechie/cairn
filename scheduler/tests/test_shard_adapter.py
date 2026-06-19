"""The real Shard fork plugs into the same BlockRuntime seam as the mock (CPU-verifiable).

The SGLang forward needs a GPU, so we don't run it here — we verify the *seam conformance*
and that the GPU path is correctly gated (raises, never silently wrong)."""

import pathlib
import sys

import pytest

# fork/ is a vendored package, not installed — put it on the path.
_FORK = pathlib.Path(__file__).resolve().parents[2] / "fork"
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

from cairn_scheduler import fit  # noqa: E402
from cairn_scheduler.runtime import BlockRuntime  # noqa: E402

from adapter import ShardBlockRuntime, build_shard_pipeline  # noqa: E402


def test_adapter_conforms_to_blockruntime_seam():
    rt = ShardBlockRuntime(0, 0, 8, "gpt-oss-120b", device="cpu")
    assert isinstance(rt, BlockRuntime)  # runtime_checkable Protocol — structural conformance
    assert (rt.stage, rt.layer_start, rt.layer_end) == (0, 0, 8)
    assert rt.kv_len("s") == 0
    rt.free_stream("s")  # no-op on an unknown stream, must not raise


def test_forward_and_load_are_gpu_gated():
    rt = ShardBlockRuntime(1, 9, 17, "gpt-oss-120b")
    with pytest.raises(NotImplementedError):
        rt.load()  # load_shard → VRAM (rung 2)
    with pytest.raises(NotImplementedError):
        rt.forward("s", 123, 0)  # SGLang forward → CUDA (rung 2)


def test_build_shard_pipeline_mirrors_fit(gpt_oss_cfg):
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    pipe = build_shard_pipeline(r, gpt_oss_cfg.hf_repo or "gpt-oss-120b")
    assert len(pipe) == r.n
    assert [s.stage for s in pipe] == list(range(r.n))
    assert all(isinstance(s, BlockRuntime) for s in pipe)
    # layer ranges line up with the fit, contiguous and inclusive
    assert pipe[0].layer_start == 0
    assert pipe[-1].layer_end == gpt_oss_cfg.num_layers - 1
