"""The §12 spec-time calcs — computations, not decisions.

These resolve at build (some inputs, e.g. exact usable VRAM, are hardware-measured;
the config carries placeholders flagged TODO). Everything here is a pure function so
it is fully unit-testable with zero GPU.

    §12.1  exact usable VRAM per card  → exact weight budget
    §12.2  KV per stream per block     → K_max per stage → real N
    §12.3  micro-batch depth           (decode-interleave; modelled in the scheduler, later)
    §12.4  warm-pool size              (observed eviction rate; v1.2 adaptive)
"""

from __future__ import annotations

from .model_config import ModelConfig, dtype_bytes


def usable_vram_bytes(cfg: ModelConfig) -> int:
    """§12.1 — card VRAM minus CUDA context + framework reserve + activation buffer.

    What's left is the budget for block weights + KV headroom. This is why "usable
    VRAM" is meaningfully less than card VRAM.
    """
    usable = cfg.gpu_vram_bytes - cfg.framework_overhead_bytes - cfg.activation_buffer_bytes
    if usable <= 0:
        raise ValueError(
            f"{cfg.name}: non-positive usable VRAM ({usable}) — overheads exceed card VRAM"
        )
    return usable


def kv_bytes_per_token_per_layer(cfg: ModelConfig) -> int:
    """KV-cache bytes for ONE token at ONE layer.

    GQA: K and V each have `num_key_value_heads * head_dim` elements per token per
    layer (the *KV* head count, not the query-head count — this is the GQA saving).
    """
    elems = 2 * cfg.num_key_value_heads * cfg.head_dim  # K + V
    return int(round(elems * dtype_bytes(cfg.kv_dtype)))


def kv_bytes_per_stream(cfg: ModelConfig, num_layers: int, context_len: int) -> int:
    """KV headroom one stream needs on a node holding `num_layers` layers.

    v1.0 `kv_model: full` treats every layer as full-attention (conservative upper
    bound). TODO swa-aware: ~half of gpt-oss's layers are sliding-window(128), so their
    per-stream KV is capped at min(context_len, sliding_window) (spec §12.2).
    """
    return kv_bytes_per_token_per_layer(cfg) * num_layers * context_len


def embedding_bytes(cfg: ModelConfig) -> int:
    """Token-embedding matrix: vocab × hidden, at the embedding dtype (not quantised)."""
    return int(round(cfg.vocab_size * cfg.hidden_size * dtype_bytes(cfg.embedding_dtype)))


def lm_head_bytes(cfg: ModelConfig) -> int:
    """Unembedding (lm_head): same shape as the embedding, or 0 if weights are tied."""
    if cfg.tie_word_embeddings:
        return 0
    return embedding_bytes(cfg)


def per_layer_weight_bytes(cfg: ModelConfig) -> int:
    """Uniform per-layer weight estimate = (total − embedding − lm_head) / num_layers.

    ASSUMPTION: layers are equal-sized. For MoE the expert-heavy layers vary; a
    per-layer override (config `footprint.per_layer_weight_bytes`) is supported by the
    schema and would feed the fit directly — not wired yet (uniform is fine for v1.0
    homogeneous placement).
    """
    layers_bytes = cfg.total_weight_bytes - embedding_bytes(cfg) - lm_head_bytes(cfg)
    if layers_bytes <= 0:
        raise ValueError(f"{cfg.name}: embedding+lm_head exceed total weight footprint")
    return layers_bytes // cfg.num_layers


def k_max_for_node(cfg: ModelConfig, num_layers: int, context_len: int, extra_bytes: int = 0) -> int:
    """§12.2 — how many streams' KV fit on a node after its weights + extras.

    `extra_bytes` = embedding and/or lm_head if this node pins them. Returns the
    VRAM-bounded ceiling K_max for this stage (spec §4.3 dial 2).
    """
    usable = usable_vram_bytes(cfg)
    weights = num_layers * per_layer_weight_bytes(cfg) + extra_bytes
    free_for_kv = usable - weights
    per_stream = kv_bytes_per_stream(cfg, num_layers, context_len)
    if per_stream <= 0:
        return 0
    if free_for_kv <= 0:
        return 0
    return free_for_kv // per_stream
