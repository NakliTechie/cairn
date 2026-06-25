"""V4 hyper-connection forward-path unit tests (CPU, no real sglang).

`SglangNodeRuntime._forward_layers` branches on model arch. The DeepSeek-V4 path (added 2026-06-25,
after the GPU run proved the LOAD path but died at the warmup forward) must honor V4's layer contract,
which differs from the residual-stream model the generic forward was proven on (L4, 2026-06-21):

  - hidden is 3D `[S, hc_mult, H]` (embed -> `unsqueeze(1).repeat(1, hc, 1)`),
  - each layer is `(positions, hidden_states, input_ids, forward_batch, input_ids_global) -> tensor`
    with the residual folded INTERNALLY (one tensor crosses the wire, no `(hidden, residual)` tuple),
  - the tail does `hc_head` (collapse the hc dim) -> single-arg `norm` -> `lm_head`.

These tests drive `_forward_layers` directly with mock V4 internals, so the tensor contract is checked
on CPU without building a real sglang ModelRunner. They DON'T prove the real V4 layers compute the
right thing (that's the GPU re-run) — they prove the fork drives them with the right shapes/args.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

_FORK = pathlib.Path(__file__).resolve().parent.parent
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

torch = pytest.importorskip("torch")

from shard.node import LayerRange  # noqa: E402
from shard.sglang_node import SglangNodeRuntime  # noqa: E402


class _FB:
    """Minimal ForwardBatch stand-in — _forward_layers only reads .input_ids and .positions."""
    def __init__(self, input_ids, positions):
        self.input_ids = input_ids
        self.positions = positions


def _make_v4_runtime(*, is_embed, is_tail, start, end, n=4, hc=2, H=8, V=16):
    """A SglangNodeRuntime wired with mock V4 internals (bypassing load_shard) so _forward_layers is
    unit-testable on CPU. The mock layer ASSERTS the 3D `[S, hc, H]` contract and returns one tensor;
    hc_head/norm/logits_processor assert/enforce the tail shape. Returns (runtime, call-recorder)."""
    calls = {"layers": [], "hc_head": 0, "norm": 0, "lp": 0}

    class _Layer:
        def __call__(self, positions, hidden_states, input_ids, forward_batch, input_ids_global):
            assert hidden_states.dim() == 3 and hidden_states.shape[1] == hc, "V4 hidden must be [S, hc, H]"
            calls["layers"].append((tuple(hidden_states.shape), int(input_ids.shape[0]),
                                    int(input_ids_global.shape[0])))
            return hidden_states + 1                          # single tensor back (residual internal)

    class _Inner:
        hc_mult = hc
        hc_head_fn = hc_head_scale = hc_head_base = None
        def __init__(self):
            self.layers = [_Layer() for _ in range(n)]
        def embed_tokens(self, ids):                         # [S] -> [S, H]
            return torch.ones(int(ids.shape[0]), H)
        def hc_head(self, hidden, fn, scale, base):          # [S, hc, H] -> [S, H]
            assert hidden.dim() == 3 and hidden.shape[1] == hc
            calls["hc_head"] += 1
            return hidden.mean(dim=1)
        def norm(self, hidden):                              # V4 norm is single-arg
            assert hidden.dim() == 2
            calls["norm"] += 1
            return hidden

    class _Result:
        def __init__(self, logits):
            self.next_token_logits = logits

    class _LP:
        # canonical DeepseekV4ForCausalLM signature: aux_hidden_states + hidden_states_before_norm
        def __call__(self, ids, hidden, lm_head, fb, aux_hidden_states=None,
                     hidden_states_before_norm=None):
            calls["lp"] += 1
            assert aux_hidden_states is None and hidden_states_before_norm is None
            return _Result(torch.zeros(int(hidden.shape[0]), V))

    class _Model:
        logits_processor = _LP()
        lm_head = object()

    class _Runner:
        model = _Model()

    rt = SglangNodeRuntime("m", LayerRange(start, end), device="cpu")
    rt._runner = _Runner()
    rt._inner = _Inner()
    rt._is_v4 = True
    rt._is_embed, rt._is_tail, rt._start, rt._end = is_embed, is_tail, start, end
    return rt, calls


def test_v4_entry_embeds_to_3d_and_hands_off():
    """Entry: embed ids -> [S, hc, H], run its slice, hand off the 3D hidden as [1, S, hc, H]."""
    rt, calls = _make_v4_runtime(is_embed=True, is_tail=False, start=0, end=2)
    ids = torch.arange(3)                                    # S = 3 tokens
    out = rt._forward_layers(x=None, fb=_FB(ids, torch.zeros(3)), s_len=3)
    assert out.shape == (1, 3, 2, 8)                         # [1, S, hc_mult, H] — preserves the hc dim
    assert [a[0] for a in calls["layers"]] == [(3, 2, 8), (3, 2, 8)]   # 2 layers, each on [S, hc, H]
    assert calls["hc_head"] == 0 and calls["norm"] == 0      # not the tail


def test_v4_middle_reshapes_wire_and_runs_slice():
    """Middle: receive [1, S, hc, H] off the wire, run only its [start,end) layers, hand off 3D again."""
    rt, calls = _make_v4_runtime(is_embed=False, is_tail=False, start=2, end=4)
    x = torch.zeros(1, 3, 2, 8)                              # wire payload [1, S, hc, H]
    out = rt._forward_layers(x=x, fb=_FB(torch.zeros(3, dtype=torch.long), torch.zeros(3)), s_len=3)
    assert out.shape == (1, 3, 2, 8)
    assert len(calls["layers"]) == 2                         # layers [2,4)


def test_v4_tail_applies_hc_head_norm_lm_head():
    """Tail: hc_head (collapse hc) -> single-arg norm -> logits_processor -> [1, S, V]."""
    rt, calls = _make_v4_runtime(is_embed=False, is_tail=True, start=2, end=4, V=16)
    x = torch.zeros(1, 3, 2, 8)
    out = rt._forward_layers(x=x, fb=_FB(torch.zeros(3, dtype=torch.long), torch.zeros(3)), s_len=3)
    assert calls["hc_head"] == 1 and calls["norm"] == 1 and calls["lp"] == 1
    assert out.shape == (1, 3, 16)                           # [1, S, V]


def test_v4_layers_receive_input_ids_and_global():
    """V4 layers get input_ids AND input_ids_global (== input_ids at tp=1/dp=1), both length S."""
    rt, calls = _make_v4_runtime(is_embed=True, is_tail=False, start=0, end=1, n=2)
    rt._forward_layers(x=None, fb=_FB(torch.arange(4), torch.zeros(4)), s_len=4)
    shp, n_ids, n_ids_global = calls["layers"][0]
    assert shp == (4, 2, 8) and n_ids == 4 and n_ids_global == 4


def test_residual_stream_path_unchanged():
    """REGRESSION: the small-model (residual-stream) branch is byte-identical to the prior forward —
    flat [S, H] hidden, (hidden, residual) layers, fold at the boundary, [1, S, H] hand-off (2D/token,
    NOT V4's 4D). Guards the L4-proven path (test_sglang_split / test_multinode_pipeline) from the refactor."""
    H = 8

    class _Layer:
        def __call__(self, positions, hidden, fb, residual):
            return hidden + 1, (residual if residual is not None else hidden)

    class _Inner:
        def __init__(self):
            self.layers = [_Layer(), _Layer()]
        def embed_tokens(self, ids):
            return torch.ones(int(ids.shape[0]), H)
        def norm(self, hidden, residual):
            return hidden, residual

    rt = SglangNodeRuntime("m", LayerRange(0, 2), device="cpu")
    rt._runner = type("R", (), {"model": object()})()
    rt._inner = _Inner()
    rt._is_v4 = False
    rt._is_embed, rt._is_tail, rt._start, rt._end = True, False, 0, 2
    out = rt._forward_layers(x=None, fb=_FB(torch.arange(3), torch.zeros(3)), s_len=3)
    assert out.shape == (1, 3, H)                            # flat hand-off, not 4D


def test_tree_cache_built_from_runner_pools_and_cached(monkeypatch):
    """`_get_tree_cache` must build a ChunkCache from the runner's pools (sglang's V4 KV-alloc path
    now calls `tree_cache.supports_swa()` UNCONDITIONALLY, so `tree_cache=None` AttributeErrors), and
    cache the instance (one tree_cache per runtime, like sglang's scheduler). Injects a fake
    `sglang.srt.mem_cache.chunk_cache` module so the construction path runs without real sglang."""
    import types

    built = []

    class _FakeChunkCache:
        def __init__(self, params):
            built.append(params)
            self.params = params

    fake_mod = types.ModuleType("sglang.srt.mem_cache.chunk_cache")
    fake_mod.ChunkCache = _FakeChunkCache
    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.srt.mem_cache.chunk_cache"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    monkeypatch.setitem(sys.modules, "sglang.srt.mem_cache.chunk_cache", fake_mod)

    rt = SglangNodeRuntime("m", LayerRange(0, 2), device="cpu")
    rt._runner = type("R", (), {
        "req_to_token_pool": "RTP", "token_to_kv_pool_allocator": "ALLOC",
        "page_size": 1, "sliding_window_size": 4096,
    })()

    tc1 = rt._get_tree_cache()
    tc2 = rt._get_tree_cache()
    assert tc1 is tc2                                        # cached — built once
    assert len(built) == 1
    p = tc1.params
    assert p.req_to_token_pool == "RTP"
    assert p.token_to_kv_pool_allocator == "ALLOC"
    assert p.page_size == 1
    assert p.sliding_window_size == 4096                    # carried for a later SWAChunkCache swap
