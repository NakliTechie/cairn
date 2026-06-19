"""The fit algorithm — spec §3.

Greedy, balanced, contiguous-layer placement that **reserves KV headroom for the
target K from day one**. This is what makes "usable VRAM" < card VRAM and couples N
back to K (more KV headroom → fewer layers per node → more stages). Embedding pins to
stage 0, lm_head to stage N−1.

The algorithm:
  1. Greedy fill: walk layers, accumulate onto the current node until the next layer
     would exceed its budget (weights + KV-for-K), then open the next node.
  2. Place lm_head on the last stage (or a thin tail stage if it won't fit).
  3. Rebalance toward minimising the max-loaded stage (the pipeline runs at the speed
     of its slowest stage). v1.0 = homogeneous pool; heterogeneous min-max is a
     documented later refinement.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .domain import BlockAssignment, FitResult
from .model_config import ModelConfig
from . import calcs


class FitError(ValueError):
    """The model cannot be placed on the pool with the requested K / context."""


def fit(
    cfg: ModelConfig,
    *,
    target_k: Optional[int] = None,
    context_len: Optional[int] = None,
    rebalance: bool = True,
) -> FitResult:
    k = target_k if target_k is not None else cfg.target_k
    ctx = context_len if context_len is not None else (cfg.fit_max_context or cfg.max_context)
    if k < 1:
        raise FitError("target_k must be >= 1")
    if ctx < 1:
        raise FitError("context_len must be >= 1")

    usable = calcs.usable_vram_bytes(cfg)
    per_layer_w = calcs.per_layer_weight_bytes(cfg)
    emb = calcs.embedding_bytes(cfg)
    lmh = calcs.lm_head_bytes(cfg)
    kv_tok_layer = calcs.kv_bytes_per_token_per_layer(cfg)

    def weights_of(nl: int, holds_emb: bool, holds_lmh: bool) -> int:
        return nl * per_layer_w + (emb if holds_emb else 0) + (lmh if holds_lmh else 0)

    def kv_of(nl: int) -> int:
        return k * ctx * kv_tok_layer * nl

    def fits(nl: int, holds_emb: bool, holds_lmh: bool) -> bool:
        return weights_of(nl, holds_emb, holds_lmh) + kv_of(nl) <= usable

    def peak(stage_layers: List[int]) -> int:
        """Max stage occupancy (weights + KV) for a layout — the pipeline's bottleneck."""
        n = len(stage_layers)
        return max(
            weights_of(nl, idx == 0, idx == n - 1) + kv_of(nl)
            for idx, nl in enumerate(stage_layers)
        )

    # --- Phase 1: greedy contiguous fill (lm_head not yet reserved) ---
    stage_layers: List[int] = []
    li = 0
    while li < cfg.num_layers:
        holds_emb = (len(stage_layers) == 0)
        count = 0
        while li + count < cfg.num_layers and fits(count + 1, holds_emb, False):
            count += 1
        if count == 0:
            extra = "embedding + " if holds_emb else ""
            raise FitError(
                f"{cfg.name}: a single layer ({extra}weights + KV for K={k} @ ctx={ctx}) "
                f"exceeds usable VRAM ({usable} B). Lower K/ctx or use a bigger card."
            )
        stage_layers.append(count)
        li += count

    # --- Phase 2: place lm_head ---
    n = len(stage_layers)
    last_holds_emb = (n == 1)
    if not fits(stage_layers[-1], last_holds_emb, True):
        # lm_head won't fit on the last layer-stage → give it a thin tail stage.
        if not fits(0, False, True):
            raise FitError(f"{cfg.name}: lm_head alone does not fit a stage")
        stage_layers.append(0)

    # --- Phase 3: rebalance toward min-max stage (homogeneous pool) ---
    if rebalance:
        stage_layers = _rebalance(stage_layers, fits, peak)

    return _build_result(cfg, stage_layers, k, ctx, usable, weights_of, kv_of)


def _rebalance(stage_layers: List[int], fits, peak) -> List[int]:
    """Even-out layers across the existing stage count, adopting the even split ONLY if
    it is feasible AND lowers the peak stage. The greedy layout is feasible by
    construction, so the result is never worse than greedy (emb/lm_head make the end
    stages heavier per layer, so a naive even split sometimes raises the peak)."""
    n = len(stage_layers)
    total = sum(stage_layers)
    has_tail = stage_layers[-1] == 0 and n > 1  # lm_head-only tail stage carries 0 layers
    layer_stages = n - 1 if has_tail else n
    if layer_stages <= 1 or total == 0:
        return stage_layers

    base, rem = divmod(total, layer_stages)
    even = [base + (1 if i < rem else 0) for i in range(layer_stages)]
    if has_tail:
        even.append(0)

    for idx, nl in enumerate(even):
        if not fits(nl, idx == 0, idx == len(even) - 1):
            return stage_layers  # even split overflows an end stage → keep greedy
    return even if peak(even) < peak(stage_layers) else stage_layers


def _build_result(cfg, stage_layers, k, ctx, usable, weights_of, kv_of) -> FitResult:
    n = len(stage_layers)
    assignments: List[BlockAssignment] = []
    cursor = 0
    max_stage_vram = 0
    for idx, nl in enumerate(stage_layers):
        holds_emb = (idx == 0)
        holds_lmh = (idx == n - 1)
        start = cursor
        end = cursor + nl - 1  # inclusive; end < start when nl == 0 (tail stage)
        est = weights_of(nl, holds_emb, holds_lmh) + kv_of(nl) + cfg.activation_buffer_bytes
        max_stage_vram = max(max_stage_vram, est)
        assignments.append(
            BlockAssignment(
                stage=idx,
                layer_start=start,
                layer_end=end,
                holds_embedding=holds_emb,
                holds_lm_head=holds_lmh,
                est_vram_bytes=est,
            )
        )
        cursor += nl

    assert cursor == cfg.num_layers, f"fit lost layers: placed {cursor}/{cfg.num_layers}"
    return FitResult(
        model_name=cfg.name,
        assignments=assignments,
        target_k=k,
        max_context=ctx,
        usable_vram_bytes=usable,
        max_stage_vram_bytes=max_stage_vram,
    )
