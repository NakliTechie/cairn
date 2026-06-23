"""Path-β loader-patch unit tests (CPU, no real sglang).

WHAT this proves (no GPU, no sglang install needed):
  (1) The SglangNodeRuntime constructor accepts and validates `cairn_pp_rank` / `cairn_pp_size`
      (from args + from env vars CAIRN_PP_{RANK,SIZE}).
  (2) The `_CairnFakePPGroup` reports the right partition for sglang's `make_layers(...)` to slice.
  (3) The patch-then-restore sequence around `ModelRunner(...)` is structurally correct — under
      a mocked sglang.srt.distributed.parallel_state module, the monkey-patch is installed before
      ModelRunner is invoked and restored after (success AND exception paths).

WHAT this does NOT prove (gated on the GPU box, future runs):
  - That sglang's real model classes (e.g. DeepSeek-V2/V3) actually pick up the fake at the
    `make_layers(pp_rank=self.pp_group.rank_in_group, ...)` call site. The code path is known
    from upstream (see plan/workplan.md Chunk C / Path C notes) but real verification requires
    a live `pp_size=N` load on the box.
  - That the loaded ModelRunner's `model.model.layers[s:e]` are real layers, with layers outside
    that slice being weight-less `PPMissingLayer` placeholders. Trivially asserted on the box via
    `type(runner.model.model.layers[i]).__name__ == "PPMissingLayer"`.
"""

from __future__ import annotations

import os
import pathlib
import sys
import types
from typing import Any, Dict

import pytest

_FORK = pathlib.Path(__file__).resolve().parent.parent
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

from shard.node import LayerRange  # noqa: E402
from shard.sglang_node import (  # noqa: E402
    SglangNodeRuntime,
    SglangNotAvailable,
    _CairnFakePPGroup,
)


# ----------------- (1) constructor + validation --------------------------------------------------


def _lr(s: int = 0, e: int = 14) -> LayerRange:
    return LayerRange(start=s, end=e)


def test_pp_args_default_to_none():
    r = SglangNodeRuntime("model", _lr(), device="cpu")
    assert r.cairn_pp_rank is None and r.cairn_pp_size is None


def test_pp_args_kwarg_pair_accepted():
    r = SglangNodeRuntime("model", _lr(), device="cpu", cairn_pp_rank=1, cairn_pp_size=3)
    assert r.cairn_pp_rank == 1 and r.cairn_pp_size == 3


def test_pp_args_env_pair_accepted(monkeypatch):
    monkeypatch.setenv("CAIRN_PP_RANK", "2")
    monkeypatch.setenv("CAIRN_PP_SIZE", "4")
    r = SglangNodeRuntime("model", _lr(), device="cpu")
    assert r.cairn_pp_rank == 2 and r.cairn_pp_size == 4


def test_pp_args_kwarg_wins_over_env(monkeypatch):
    monkeypatch.setenv("CAIRN_PP_RANK", "9")
    monkeypatch.setenv("CAIRN_PP_SIZE", "9")
    r = SglangNodeRuntime("model", _lr(), device="cpu", cairn_pp_rank=0, cairn_pp_size=2)
    assert r.cairn_pp_rank == 0 and r.cairn_pp_size == 2


def test_pp_args_lone_rank_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        SglangNodeRuntime("model", _lr(), device="cpu", cairn_pp_rank=0)


def test_pp_args_lone_size_rejected():
    with pytest.raises(ValueError, match="must be set together"):
        SglangNodeRuntime("model", _lr(), device="cpu", cairn_pp_size=3)


def test_pp_args_rank_out_of_range_rejected():
    with pytest.raises(ValueError, match=r"cairn_pp_rank \(3\) must be in"):
        SglangNodeRuntime("model", _lr(), device="cpu", cairn_pp_rank=3, cairn_pp_size=3)


def test_pp_args_negative_rank_rejected():
    with pytest.raises(ValueError, match=r"cairn_pp_rank \(-1\) must be in"):
        SglangNodeRuntime("model", _lr(), device="cpu", cairn_pp_rank=-1, cairn_pp_size=3)


# ----------------- (2) the fake pp group has the right shape ------------------------------------


def test_fake_group_reports_rank_and_size():
    g = _CairnFakePPGroup(rank=1, size=3)
    assert g.rank_in_group == 1
    assert g.world_size == 3
    assert g.ranks == [0, 1, 2]


def test_fake_group_first_last_rank_flags():
    head = _CairnFakePPGroup(0, 3)
    mid = _CairnFakePPGroup(1, 3)
    tail = _CairnFakePPGroup(2, 3)
    assert head.is_first_rank and not head.is_last_rank
    assert not mid.is_first_rank and not mid.is_last_rank
    assert not tail.is_first_rank and tail.is_last_rank


def test_fake_group_next_prev_wrap_correctly():
    g = _CairnFakePPGroup(1, 3)
    assert g.next_rank == 2 and g.prev_rank == 0
    head = _CairnFakePPGroup(0, 3)
    assert head.next_rank == 1 and head.prev_rank == 2  # ring topology, fine


def test_fake_group_collectives_refuse_loudly():
    # If anything mistakenly tries a real PP collective (it shouldn't — Cairn drives layers
    # directly), it must crash loudly rather than hang on NCCL after a node vanishes.
    g = _CairnFakePPGroup(0, 3)
    import torch  # collective signatures want a tensor, but the fake should reject earlier
    t = None
    for method in ("all_reduce", "all_gather", "broadcast", "send", "recv"):
        with pytest.raises(RuntimeError, match="NCCL inter-stage collectives must never fire"):
            getattr(g, method)(t)
    g.barrier()  # barrier is benign — some init paths call it; must not raise


# ----------------- (3) monkey-patch composes (mocked sglang) -----------------------------------


def _install_fake_sglang(monkeypatch, num_layers: int):
    """Wire up the minimum sglang surface SglangNodeRuntime.load_shard touches, so we can
    drive the patch-then-restore sequence without needing real sglang."""

    # The fake ModelRunner records get_pp_group() AT THE MOMENT it's constructed (proxy for
    # "the model class would call make_layers(pp_rank=get_pp_group().rank_in_group, ...) at
    # load time"), so the test can assert the monkey-patch was IN PLACE during construction.
    seen: Dict[str, Any] = {"pp_group_at_init": None, "ctor_called": False}

    class _FakeLayer:
        pass

    class _FakeInner:
        def __init__(self) -> None:
            self.layers = [_FakeLayer() for _ in range(num_layers)]

    class _FakeModel:
        def __init__(self) -> None:
            self.model = _FakeInner()

    class _FakeServerArgs:
        def __init__(self, **kw) -> None:
            self.__dict__.update(kw)

    class _FakeModelConfig:
        @classmethod
        def from_server_args(cls, sa):
            return cls()

    class _FakeModelRunner:
        def __init__(self, *a, **kw) -> None:
            seen["ctor_called"] = True
            # Snapshot the monkey-patch state at the very moment ModelRunner runs.
            seen["pp_group_at_init"] = parallel_state.get_pp_group()
            self.model = _FakeModel()

    # Build the module tree: sglang.srt.{server_args, configs.model_config, model_executor.model_runner, distributed}
    sglang = types.ModuleType("sglang")
    srt = types.ModuleType("sglang.srt")
    server_args = types.ModuleType("sglang.srt.server_args")
    configs = types.ModuleType("sglang.srt.configs")
    model_config_mod = types.ModuleType("sglang.srt.configs.model_config")
    model_executor = types.ModuleType("sglang.srt.model_executor")
    model_runner_mod = types.ModuleType("sglang.srt.model_executor.model_runner")
    distributed = types.ModuleType("sglang.srt.distributed")
    parallel_state = types.ModuleType("sglang.srt.distributed.parallel_state")

    server_args.ServerArgs = _FakeServerArgs
    model_config_mod.ModelConfig = _FakeModelConfig
    model_runner_mod.ModelRunner = _FakeModelRunner

    # Default get_pp_group returns the "real" (pp_size=1) group; the runtime patches this in.
    _baseline = _CairnFakePPGroup(0, 1)
    parallel_state.get_pp_group = lambda: _baseline

    # get_pp_indices: even balanced split (mirrors sglang's upstream behavior closely enough for
    # the partition-mismatch assertion test).
    def _get_pp_indices(num_layers: int, rank: int, size: int):
        per = num_layers // size
        rem = num_layers % size
        start = rank * per + min(rank, rem)
        end = start + per + (1 if rank < rem else 0)
        return start, end
    distributed.get_pp_indices = _get_pp_indices

    monkeypatch.setitem(sys.modules, "sglang", sglang)
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    monkeypatch.setitem(sys.modules, "sglang.srt.server_args", server_args)
    monkeypatch.setitem(sys.modules, "sglang.srt.configs", configs)
    monkeypatch.setitem(sys.modules, "sglang.srt.configs.model_config", model_config_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor", model_executor)
    monkeypatch.setitem(sys.modules, "sglang.srt.model_executor.model_runner", model_runner_mod)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed", distributed)
    monkeypatch.setitem(sys.modules, "sglang.srt.distributed.parallel_state", parallel_state)

    return seen, parallel_state, _baseline


def test_load_shard_patches_get_pp_group_during_modelrunner_init(monkeypatch):
    """The monkey-patch must be IN PLACE while ModelRunner runs (so sglang's make_layers sees
    the fake), and RESTORED after."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    seen, pstate, baseline = _install_fake_sglang(monkeypatch, num_layers=43)
    # N=3 split, this is the middle stage; partition for 43 layers, rank 1 of 3 = [15, 29).
    r = SglangNodeRuntime("model", _lr(15, 29), device="cpu",
                         cairn_pp_rank=1, cairn_pp_size=3)
    r.load_shard()
    # At ModelRunner construction, the patched fake was active:
    assert seen["ctor_called"]
    captured = seen["pp_group_at_init"]
    assert isinstance(captured, _CairnFakePPGroup)
    assert captured.rank_in_group == 1 and captured.world_size == 3
    # After load_shard, the baseline get_pp_group is restored:
    assert pstate.get_pp_group() is baseline


def test_load_shard_restores_get_pp_group_on_exception(monkeypatch):
    """If ModelRunner raises, the restore in `finally` must still run — else a subsequent load
    in the same process would see Cairn's fake leak into unrelated code paths."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    seen, pstate, baseline = _install_fake_sglang(monkeypatch, num_layers=43)

    class _Boom(RuntimeError):
        pass

    def _raises(*a, **kw):
        raise _Boom("simulated load failure")
    sys.modules["sglang.srt.model_executor.model_runner"].ModelRunner = _raises

    r = SglangNodeRuntime("model", _lr(0, 14), device="cpu",
                         cairn_pp_rank=0, cairn_pp_size=3)
    with pytest.raises(_Boom):
        r.load_shard()
    assert pstate.get_pp_group() is baseline


def test_load_shard_rejects_layer_range_mismatched_with_pp_partition(monkeypatch):
    """layer_range MUST equal sglang's pp partition; mismatch would route the forward through
    PPMissingLayer placeholders (zero weights). Catch this loudly at load time, not at decode."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    _install_fake_sglang(monkeypatch, num_layers=43)
    # Rank-1 of 3 in a 43-layer model partitions to [15, 29); claiming [10, 25) is a mismatch.
    r = SglangNodeRuntime("model", _lr(10, 25), device="cpu",
                         cairn_pp_rank=1, cairn_pp_size=3)
    with pytest.raises(ValueError, match="does not match sglang pp partition"):
        r.load_shard()


def test_load_shard_pp_size_1_skips_patching(monkeypatch):
    """pp_size=1 path is the existing behavior — no monkey-patch, no partition assertion. Used
    by every cheap-model multi-stream test today (fork/tests/test_multistream.py)."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    seen, pstate, baseline = _install_fake_sglang(monkeypatch, num_layers=43)
    r = SglangNodeRuntime("model", _lr(0, 43), device="cpu")  # no pp args → existing behavior
    r.load_shard()
    # ModelRunner ran, but with the BASELINE pp_group (Cairn never patched):
    assert seen["pp_group_at_init"] is baseline


def test_pp_size_in_shared_cache_key(monkeypatch):
    """Two runtimes in the same process with DIFFERENT cairn_pp configs must NOT share a runner
    (each holds a different layer slice). Otherwise the second would forward through the wrong
    slice's weights."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    _install_fake_sglang(monkeypatch, num_layers=43)
    SglangNodeRuntime._SHARED.clear()
    # 43 layers, even split with remainder front-loaded: rank 0 -> [0,15), rank 1 -> [15,29), rank 2 -> [29,43).
    a = SglangNodeRuntime("m", _lr(0, 15), device="cpu", cairn_pp_rank=0, cairn_pp_size=3)
    b = SglangNodeRuntime("m", _lr(15, 29), device="cpu", cairn_pp_rank=1, cairn_pp_size=3)
    a.load_shard()
    b.load_shard()
    assert a._runner is not b._runner   # different keys → different runners
    SglangNodeRuntime._SHARED.clear()
