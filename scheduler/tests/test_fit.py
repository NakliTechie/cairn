import dataclasses

import pytest

from cairn_scheduler import fit
from cairn_scheduler.fit import FitError


def _all_layers_covered(result, num_layers):
    """Stages must tile layers 0..num_layers-1 contiguously, no gaps/overlaps."""
    covered = []
    for a in result.assignments:
        covered.extend(range(a.layer_start, a.layer_end + 1))
    return covered == list(range(num_layers))


def test_fit_places_every_layer(model_cfg):
    r = fit(model_cfg)
    assert r.n >= 1
    assert _all_layers_covered(r, model_cfg.num_layers)
    assert sum(a.num_layers for a in r.assignments) == model_cfg.num_layers


def test_embedding_stage0_lm_head_last(model_cfg):
    r = fit(model_cfg)
    assert r.assignments[0].holds_embedding
    assert r.assignments[-1].holds_lm_head
    # embedding and lm_head pinned to exactly one stage each.
    assert sum(a.holds_embedding for a in r.assignments) == 1
    assert sum(a.holds_lm_head for a in r.assignments) == 1


def test_every_stage_fits_card(model_cfg):
    r = fit(model_cfg)
    for a in r.assignments:
        # weights + KV (est minus the activation buffer) must sit within usable VRAM.
        assert a.est_vram_bytes - model_cfg.activation_buffer_bytes <= r.usable_vram_bytes
        assert a.est_vram_bytes <= model_cfg.gpu_vram_bytes


def test_real_n_grows_with_k(model_cfg):
    """Spec §3: reserving KV headroom for a larger K makes the real N larger than the
    weights-only estimate (fewer layers fit per node)."""
    n_small = fit(model_cfg, target_k=1, context_len=8192).n
    n_large = fit(model_cfg, target_k=64, context_len=8192).n
    assert n_large > n_small


def test_infeasible_raises(model_cfg):
    # Shrink the card so even a single-layer stage (+ its pinned embedding) can't fit → FitError,
    # not a bad plan. llama's embedding alone is ~1 GiB, so a 0.5 GiB card is infeasible.
    tiny = dataclasses.replace(
        model_cfg,
        gpu_vram_bytes=512 * 1024**2,
        framework_overhead_bytes=64 * 1024**2,
        activation_buffer_bytes=32 * 1024**2,
    )
    with pytest.raises(FitError):
        fit(tiny, target_k=1, context_len=1024)


def test_rebalance_no_worse_than_greedy(model_cfg):
    balanced = fit(model_cfg, rebalance=True)
    greedy = fit(model_cfg, rebalance=False)
    # Same model placed with the same N; rebalancing must not raise the peak stage.
    assert balanced.n == greedy.n
    assert balanced.max_stage_vram_bytes <= greedy.max_stage_vram_bytes
