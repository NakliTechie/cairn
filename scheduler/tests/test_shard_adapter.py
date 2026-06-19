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


def test_sglang_runtime_structural_and_gpu_gated():
    from shard.node import LayerRange
    from shard.sglang_node import SglangNodeRuntime

    rt = SglangNodeRuntime("gpt-oss-120b", LayerRange(0, 9), device="cuda:0")
    hb = rt.heartbeat()                      # heartbeat works without CUDA (reports not-alive)
    assert hb["alive"] is False and hb["layers"] == [0, 9]
    assert rt.kv_tokens("s") == 0
    rt.free_seq("s")                          # no-op, must not raise
    with pytest.raises(Exception):            # load_shard needs a GPU/sglang (SglangNotAvailable/NotImplemented)
        rt.load_shard()
    # L1: a malformed-but-colon-bearing device string must not crash the constructor.
    for dev in ("cuda:", "cuda:abc", "cuda", "cpu"):
        assert SglangNodeRuntime("m", LayerRange(0, 1), device=dev).device_index == 0


def test_adapter_free_stream_releases_node_kv():
    """S11: ShardBlockRuntime.free_stream delegates to the backing runtime's free_seq."""
    from shard.sglang_node import SglangNodeRuntime

    rt = ShardBlockRuntime(0, 0, 8, "gpt-oss-120b", runtime_cls=SglangNodeRuntime)
    rt._node._kv_seqs["s"] = 3            # simulate KV cached on the node for stream "s"
    rt.free_stream("s")
    assert rt._node.kv_tokens("s") == 0  # released via free_seq, not just the adapter's counter


def test_adapter_accepts_sglang_runtime_cls():
    from shard.sglang_node import SglangNodeRuntime

    rt = ShardBlockRuntime(0, 0, 8, "gpt-oss-120b", runtime_cls=SglangNodeRuntime)
    assert isinstance(rt, BlockRuntime)       # still conforms to the seam with the real backend
    with pytest.raises(Exception):
        rt.load()                              # GPU-gated


def test_build_shard_pipeline_mirrors_fit(gpt_oss_cfg):
    r = fit(gpt_oss_cfg, target_k=8, context_len=4096)
    pipe = build_shard_pipeline(r, gpt_oss_cfg.hf_repo or "gpt-oss-120b")
    assert len(pipe) == r.n
    assert [s.stage for s in pipe] == list(range(r.n))
    assert all(isinstance(s, BlockRuntime) for s in pipe)
    # layer ranges line up with the fit, contiguous and inclusive
    assert pipe[0].layer_start == 0
    assert pipe[-1].layer_end == gpt_oss_cfg.num_layers - 1
