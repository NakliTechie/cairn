"""Path-β loader-patch unit tests (CPU, no real sglang).

Mechanism (rewritten 2026-06-24 after the maiden-run OOM): the OLD loader patched
`parallel_state.get_pp_group` BEFORE `ModelRunner(...)`, but `ModelRunner.__init__` runs
`init_torch_distributed()` → `initialize_model_parallel(pp_size=1)` which REBUILDS `_PP` at
world_size=1 before the model is built — discarding the patch. Every box loaded the full model
→ identical 94 GiB OOM. The 17 old tests passed because the mock recorded get_pp_group at
ModelRunner.__init__ and never modeled that re-init. The fix wraps `ModelRunner.load_model`
(runs AFTER the re-init) to mutate `_PP` to (k, N), + pins SGLANG_PP_LAYER_PARTITION.

WHAT this proves (no GPU, no sglang install needed):
  (1) The constructor accepts/validates `cairn_pp_rank`/`cairn_pp_size`/`cairn_pp_partition`
      (from args + env CAIRN_PP_{RANK,SIZE,LAYER_PARTITION}).
  (2) `_CairnFakePPGroup` reports the right partition identity; `_set_pp_identity` mutates it.
  (3) The mock NOW models the real boundary — `_FakeModelRunner.__init__` rebuilds `_PP` to (0,1),
      then `load_model()` reads get_pp_group(). The regression test asserts the model sees (k, N)
      there (so the load_model wrapper re-applied it post-re-init), and the wrapper is restored on
      success AND exception. An explicit partition pins SGLANG_PP_LAYER_PARTITION; mismatch is rejected.

WHAT this does NOT prove (gated on the GPU box, future runs):
  - That `ModelRunner.load_model` is the real post-dist-init hook point and that the real DeepSeek
    model reads get_pp_group() after it (highly likely per the sglang source dig, 2026-06-24).
  - That the loaded `model.model.layers[s:e]` are real layers with the rest weight-less
    `PPMissingLayer`, and per-box VRAM ≈ slice-sized not full. Assert on the box via
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
    """Wire up the minimum sglang surface SglangNodeRuntime.load_shard touches — and crucially,
    MODEL THE REAL BOUNDARY the 2026-06-24 maiden run exposed.

    The OLD mock recorded get_pp_group() at ModelRunner.__init__ and never modeled
    `init_torch_distributed()` rebuilding `_PP` at world_size=1 — so a get_pp_group patch that the
    real ModelRunner silently clobbers passed green here while OOMing on the box. This mock now:
      • exposes a mutable `parallel_state._PP` global; `get_pp_group()` reads it LIVE (like real sglang),
      • `_FakeModelRunner.__init__` REBUILDS `_PP` to (0,1) — the dist re-init that discards any
        pre-ModelRunner patch — BEFORE building the model,
      • the model reads get_pp_group() inside `load_model()` (where DeepseekV*Model.__init__ does),
      • `get_pp_indices` honors `SGLANG_PP_LAYER_PARTITION` (explicit) else even-split remainder-LAST
        (matching real sglang's source — the old mock was remainder-first, another inaccuracy).
    So the loader's fix (mutate `_PP` from a `load_model` wrapper) is what makes the slice correct.
    """
    # isolate process-global env the loader reads/sets
    for _k in ("SGLANG_PP_LAYER_PARTITION", "CAIRN_PP_LAYER_PARTITION", "CAIRN_PP_RANK", "CAIRN_PP_SIZE"):
        monkeypatch.delenv(_k, raising=False)

    seen: Dict[str, Any] = {"ctor_called": False, "load_model_called": False,
                            "pp_group_at_load": None, "rank_at_load": None, "size_at_load": None}

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
            # init_torch_distributed() → initialize_model_parallel(pp_size=1): the real dist re-init
            # that REBUILDS _PP at world_size=1, discarding any patch applied before ModelRunner.
            parallel_state._PP = _CairnFakePPGroup(0, 1)
            self.load_model()                 # model construction reads get_pp_group() in here
            self.model = _FakeModel()

        def load_model(self, *a, **kw):
            grp = parallel_state.get_pp_group()
            seen["load_model_called"] = True
            seen["pp_group_at_load"] = grp
            seen["rank_at_load"] = grp.rank_in_group
            seen["size_at_load"] = grp.world_size

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

    # _PP is a live module global; get_pp_group() dereferences it on every call (like real sglang).
    parallel_state._PP = _CairnFakePPGroup(0, 1)
    parallel_state.get_pp_group = lambda: parallel_state._PP

    def _get_pp_indices(n: int, rank: int, size: int):
        env = os.environ.get("SGLANG_PP_LAYER_PARTITION")
        if env:
            parts = [int(x) for x in env.split(",")]
            start = sum(parts[:rank])
            return start, start + parts[rank]
        base, rem = n // size, n % size            # even split, remainder-LAST (real sglang)
        if rank >= size - rem:
            start = rank * (base + 1) - (size - rem)
            return start, start + base + 1
        start = rank * base
        return start, start + base
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

    return seen, parallel_state, model_runner_mod


def test_load_shard_slices_via_load_model_after_dist_reinit(monkeypatch):
    """REGRESSION for the maiden-run OOM: even though ModelRunner rebuilds _PP to (0,1) during
    __init__, the model (in load_model) must read get_pp_group() == (k, N) — i.e. the loader's
    load_model wrapper re-applies the identity AFTER the dist re-init. The OLD get_pp_group patch
    (before ModelRunner) would be discarded here → this test would fail with it."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    seen, pstate, mr_mod = _install_fake_sglang(monkeypatch, num_layers=43)
    # Front-loaded split [15,14,14]; this is the middle stage (rank 1 → [15,29)).
    r = SglangNodeRuntime("model", _lr(15, 29), device="cpu",
                         cairn_pp_rank=1, cairn_pp_size=3, cairn_pp_partition=[15, 14, 14])
    r.load_shard()
    assert seen["ctor_called"] and seen["load_model_called"]
    # The model, reading get_pp_group() during load, saw OUR identity (1, 3) — not the rebuilt (0, 1):
    assert seen["rank_at_load"] == 1 and seen["size_at_load"] == 3
    # And the explicit partition was pinned for sglang's get_pp_indices:
    assert os.environ.get("SGLANG_PP_LAYER_PARTITION") == "15,14,14"
    # load_model wrapper restored after the load (no leak to other ModelRunner constructions):
    assert mr_mod.ModelRunner.load_model.__name__ != "_cairn_load_model"


def test_load_shard_restores_load_model_on_exception(monkeypatch):
    """If ModelRunner raises, the `finally` must restore ModelRunner.load_model — else the wrapper
    leaks into unrelated loads in the same process."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    seen, pstate, mr_mod = _install_fake_sglang(monkeypatch, num_layers=43)
    orig_load_model = mr_mod.ModelRunner.load_model

    class _Boom(RuntimeError):
        pass

    # A ModelRunner whose ctor blows up (after the loader has wrapped its load_model). Keeps a real
    # load_model so the loader's hasattr check + wrap/restore path is exercised around the failure.
    mr_mod.ModelRunner = type("BoomRunner", (), {
        "load_model": orig_load_model,
        "__init__": lambda self, *a, **k: (_ for _ in ()).throw(_Boom("boom")),
    })

    r = SglangNodeRuntime("model", _lr(0, 15), device="cpu",
                         cairn_pp_rank=0, cairn_pp_size=3, cairn_pp_partition=[15, 14, 14])
    with pytest.raises(_Boom):
        r.load_shard()
    # the wrapper was removed from the class even though the ctor raised:
    assert mr_mod.ModelRunner.load_model.__name__ != "_cairn_load_model"


def test_load_shard_pp_partition_env_pins_slice(monkeypatch):
    """An explicit cairn_pp_partition sets SGLANG_PP_LAYER_PARTITION so sglang slices to Cairn's
    boundaries (front-loaded), not its even-split default — and matches a front-loaded layer_range."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    _install_fake_sglang(monkeypatch, num_layers=43)
    SglangNodeRuntime._SHARED.clear()
    # rank 2, partition [15,14,14] → slice [29,43). (Even-split default would be [28,43) → mismatch.)
    r = SglangNodeRuntime("m", _lr(29, 43), device="cpu",
                         cairn_pp_rank=2, cairn_pp_size=3, cairn_pp_partition=[15, 14, 14])
    r.load_shard()  # no ValueError → the env-pinned partition matched layer_range
    assert os.environ.get("SGLANG_PP_LAYER_PARTITION") == "15,14,14"
    SglangNodeRuntime._SHARED.clear()


def test_load_shard_rejects_partition_wrong_length(monkeypatch):
    """cairn_pp_partition must have exactly cairn_pp_size entries (one layer-count per stage)."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    _install_fake_sglang(monkeypatch, num_layers=43)
    r = SglangNodeRuntime("m", _lr(0, 15), device="cpu",
                         cairn_pp_rank=0, cairn_pp_size=3, cairn_pp_partition=[15, 28])  # 2 != 3
    with pytest.raises(ValueError, match="must have cairn_pp_size=3"):
        r.load_shard()


def test_load_shard_rejects_layer_range_mismatched_with_pp_partition(monkeypatch):
    """layer_range MUST equal sglang's pp partition; mismatch would route the forward through
    PPMissingLayer placeholders (zero weights). Catch this loudly at load time, not at decode."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    _install_fake_sglang(monkeypatch, num_layers=43)
    # Pinned partition [15,14,14] → rank-1 slice is [15,29); claiming [10,25) is a mismatch.
    r = SglangNodeRuntime("model", _lr(10, 25), device="cpu",
                         cairn_pp_rank=1, cairn_pp_size=3, cairn_pp_partition=[15, 14, 14])
    with pytest.raises(ValueError, match="does not match sglang pp partition"):
        r.load_shard()


def test_load_shard_pp_size_1_skips_patching(monkeypatch):
    """pp_size=1 path is the existing behavior — no load_model wrap, no partition assertion. Used
    by every cheap-model multi-stream test today (fork/tests/test_multistream.py)."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    seen, pstate, mr_mod = _install_fake_sglang(monkeypatch, num_layers=43)
    r = SglangNodeRuntime("model", _lr(0, 43), device="cpu")  # no pp args → existing behavior
    r.load_shard()
    # ModelRunner ran, and the model saw the un-touched (rebuilt) pp_group — world_size 1, no slice:
    assert seen["size_at_load"] == 1 and seen["rank_at_load"] == 0


def test_pp_size_in_shared_cache_key(monkeypatch):
    """Two runtimes in the same process with DIFFERENT cairn_pp configs must NOT share a runner
    (each holds a different layer slice). Otherwise the second would forward through the wrong
    slice's weights."""
    monkeypatch.setattr("shard.sglang_node._HAS_TORCH", True)
    _install_fake_sglang(monkeypatch, num_layers=43)
    SglangNodeRuntime._SHARED.clear()
    # Front-loaded partition [15,14,14]: rank 0 → [0,15), rank 1 → [15,29).
    a = SglangNodeRuntime("m", _lr(0, 15), device="cpu", cairn_pp_rank=0, cairn_pp_size=3, cairn_pp_partition=[15, 14, 14])
    b = SglangNodeRuntime("m", _lr(15, 29), device="cpu", cairn_pp_rank=1, cairn_pp_size=3, cairn_pp_partition=[15, 14, 14])
    a.load_shard()
    b.load_shard()
    assert a._runner is not b._runner   # different keys → different runners
    SglangNodeRuntime._SHARED.clear()


# ----------------- (4) load-time norm-gate fix (V4 per-layer compressor.norm) --------------------
#
# REGRESSION for the 2026-06-25 box failure: slicing engaged (the maiden-run OOM was gone), but the
# run died one step later at sglang's strict weight-init check — `model.layers.{29..42}.self_attn.
# {compressor,indexer.compressor}.norm.weight` + `model.norm.weight` "not initialized". Root cause:
# deepseek_v4.py load_weights skips embed weights when `not pp_group.is_first_rank` and skips ANY
# `.norm.` weight when `not pp_group.is_last_rank`. Those are @property values derived from the
# GLOBAL torch rank (0) — NOT the rank_in_group our slicing sets — so is_last_rank is False on every
# stage, dropping V4's per-layer compressor.norm (which also matches `.norm.`). The fix forces both
# gates open for the load via a pp_group proxy (safe: embed/norm/lm_head are built unconditionally,
# so loading them on every stage is memory-neutral; the LAYER-slice gate uses start/end_layer).


def _install_fake_v4_model(monkeypatch, *, is_last_rank: bool, is_first_rank: bool = True):
    """Install a fake `sglang.srt.models.deepseek_v4` whose DeepseekV4ForCausalLM.load_weights
    replicates the real per-stage skip gate + strict check (deepseek_v4.py ~L1814/1818/1934).
    is_first/is_last_rank are @property on pp_group (as in the real GroupCoordinator) so they can't
    be setattr'd — only the load-time proxy can override them. Returns (model class, loaded-name set)."""
    loaded: set = set()

    class _PPGroup:
        world_size = 3            # plain attrs the proxy must DELEGATE (the layer-slice gate reads them)
        rank_in_group = 1
        @property
        def is_first_rank(self):  # noqa: ANN001 — derived like the real GroupCoordinator; not setattr-able
            return is_first_rank
        @property
        def is_last_rank(self):
            return is_last_rank

    class DeepseekV4ForCausalLM:
        def __init__(self):
            self.pp_group = _PPGroup()
        def load_weights(self, weights):
            loaded.clear()
            params = set(weights)                         # stand-in for params_dict (all are real params)
            for name in weights:
                if ".embed_tokens." in name and not self.pp_group.is_first_rank:
                    continue
                if ".norm." in name and not self.pp_group.is_last_rank:   # the over-broad gate
                    continue
                loaded.add(name)
            unloaded = params - loaded
            if unloaded:                                  # mirrors deepseek_v4.py:1934
                raise RuntimeError(f"Some weights are not initialized from checkpoints: {unloaded}")
            return loaded

    for modname in ("sglang", "sglang.srt", "sglang.srt.models"):
        monkeypatch.setitem(sys.modules, modname, sys.modules.get(modname) or types.ModuleType(modname))
    mod = types.ModuleType("sglang.srt.models.deepseek_v4")
    mod.DeepseekV4ForCausalLM = DeepseekV4ForCausalLM
    monkeypatch.setitem(sys.modules, "sglang.srt.models.deepseek_v4", mod)
    return DeepseekV4ForCausalLM, loaded


# the weights a non-tail stage must load: V4's per-layer sparse-attn norms + a non-norm control.
_STAGE_WEIGHTS = [
    "model.layers.29.self_attn.compressor.norm.weight",
    "model.layers.29.self_attn.indexer.compressor.norm.weight",
    "model.norm.weight",
    "model.layers.29.self_attn.wkv.weight",     # non-norm control — always loads
]


def test_load_weights_gate_drops_norms_without_fix(monkeypatch):
    """CONTROL — reproduce the box failure: with is_last_rank False the gate drops EVERY `.norm.`
    weight (incl. the per-layer compressor.norm) → the strict check raises 'not initialized'."""
    cls, _ = _install_fake_v4_model(monkeypatch, is_last_rank=False)
    with pytest.raises(RuntimeError, match="not initialized from checkpoints") as ei:
        cls().load_weights(_STAGE_WEIGHTS)
    assert "compressor.norm.weight" in str(ei.value)     # the V4 per-layer norm was dropped


def test_load_weights_rank_fix_loads_norms(monkeypatch):
    """THE FIX — after _install_load_weights_rank_fix(), load_weights views pp_group as all-ranks-
    True, so the per-layer compressor.norm/indexer.compressor.norm (+ final norm) load even on a
    stage whose real is_last_rank is False. No strict-check raise."""
    from shard.sglang_node import _install_load_weights_rank_fix
    cls, loaded = _install_fake_v4_model(monkeypatch, is_last_rank=False)
    _install_load_weights_rank_fix()
    cls().load_weights(_STAGE_WEIGHTS)                    # must NOT raise
    for n in _STAGE_WEIGHTS:
        assert n in loaded, f"{n} should load with the fix"


def test_load_weights_rank_fix_restores_pp_group(monkeypatch):
    """The proxy is installed only for the call — pp_group must be the real group again afterward
    (no leak: post-load code reading is_last_rank must see the true value)."""
    from shard.sglang_node import _install_load_weights_rank_fix, _LoadTimeAllRanksPPGroup
    cls, _ = _install_fake_v4_model(monkeypatch, is_last_rank=False)
    _install_load_weights_rank_fix()
    m = cls()
    real = m.pp_group
    m.load_weights(["model.norm.weight"])
    assert m.pp_group is real and not isinstance(m.pp_group, _LoadTimeAllRanksPPGroup)
    assert m.pp_group.is_last_rank is False              # true value restored


def test_load_weights_rank_fix_idempotent(monkeypatch):
    """Installing twice must not double-wrap (guard flag) — else nested proxies / leaks."""
    from shard.sglang_node import _install_load_weights_rank_fix
    cls, _ = _install_fake_v4_model(monkeypatch, is_last_rank=False)
    _install_load_weights_rank_fix()
    once = cls.load_weights
    _install_load_weights_rank_fix()
    assert cls.load_weights is once                      # second install is a no-op


def test_load_weights_fix_preserves_tail_and_delegates(monkeypatch):
    """Sanity: on the true tail (is_last_rank True) the fix is harmless (norms already load), and the
    proxy DELEGATES non-rank attrs (world_size/rank_in_group) so the layer-slice gate is unaffected."""
    from shard.sglang_node import _install_load_weights_rank_fix, _LoadTimeAllRanksPPGroup
    cls, loaded = _install_fake_v4_model(monkeypatch, is_last_rank=True)
    _install_load_weights_rank_fix()
    cls().load_weights(_STAGE_WEIGHTS)
    assert all(n in loaded for n in _STAGE_WEIGHTS)
    class _Real:
        world_size = 3
        rank_in_group = 2
        @property
        def is_last_rank(self): return False
        @property
        def is_first_rank(self): return False
    proxy = _LoadTimeAllRanksPPGroup(_Real())
    assert proxy.is_first_rank is True and proxy.is_last_rank is True
    assert proxy.world_size == 3 and proxy.rank_in_group == 2   # delegated to the real group
