"""Model-as-config (spec invariant #3).

A model is a YAML file in `configs/`. This module parses + validates it into a
`ModelConfig`. No model-specific code lives anywhere else — the fit and calcs read
only these fields. Adding a model = adding a YAML, never a code path.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

# Bytes-per-element for the dtypes we reference in fit/KV math.
_DTYPE_BYTES = {
    "float32": 4, "fp32": 4,
    "bfloat16": 2, "bf16": 2,
    "float16": 2, "fp16": 2,
    "float8": 1, "fp8": 1, "e4m3": 1, "e5m2": 1,
    "int8": 1,
    "int4": 1,  # packed nibble; treated as 0.5 where it matters, but KV is never int4 here
}


def dtype_bytes(name: str) -> float:
    key = name.lower()
    if key in ("int4", "nf4", "mxfp4"):
        return 0.5
    if key not in _DTYPE_BYTES:
        raise ConfigError(f"unknown dtype {name!r}")
    return float(_DTYPE_BYTES[key])


class ConfigError(ValueError):
    """Raised when a model config is missing required fields or is internally inconsistent."""


@dataclass(frozen=True)
class ModelConfig:
    name: str
    # arch
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int
    max_context: int
    tie_word_embeddings: bool
    # MoE (0 experts ⇒ dense)
    num_local_experts: int
    num_experts_per_tok: int
    sliding_window: Optional[int]
    # precision / quant
    quant_method: str
    kv_dtype: str
    embedding_dtype: str
    kv_model: str  # "full" | "swa-aware"
    # footprint
    total_weight_bytes: int
    # §12 calc inputs (hardware-measured at build)
    framework_overhead_bytes: int
    activation_buffer_bytes: int
    # fit defaults
    target_k: int
    fit_max_context: int
    # pool
    instance_type: str
    gpu: str
    gpu_vram_bytes: int
    gpu_count_per_node: int
    # provenance
    license: str = "unknown"
    hf_repo: str = ""

    @property
    def is_moe(self) -> bool:
        return self.num_local_experts > 0

    def __post_init__(self) -> None:
        if self.num_layers <= 0:
            raise ConfigError(f"{self.name}: num_layers must be > 0")
        if self.num_key_value_heads <= 0 or self.head_dim <= 0:
            raise ConfigError(f"{self.name}: GQA dims (num_key_value_heads, head_dim) must be > 0")
        # L6: a 0 here silently understates the footprint (e.g. embedding_bytes==0) and skews the fit.
        for fld in ("hidden_size", "vocab_size", "num_attention_heads", "max_context"):
            if getattr(self, fld) <= 0:
                raise ConfigError(f"{self.name}: {fld} must be > 0")
        if self.total_weight_bytes <= 0:
            raise ConfigError(f"{self.name}: footprint.total_weight_bytes must be > 0")
        if self.gpu_vram_bytes <= 0:
            raise ConfigError(f"{self.name}: pool.gpu_vram_bytes must be > 0")
        if self.target_k < 1:
            raise ConfigError(f"{self.name}: fit_defaults.target_k must be >= 1")
        if self.kv_model not in ("full", "swa-aware"):
            raise ConfigError(f"{self.name}: precision.kv_model must be 'full' or 'swa-aware'")
        # validate dtypes parse
        dtype_bytes(self.kv_dtype)
        dtype_bytes(self.embedding_dtype)


def _require(d: dict, *path):
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            raise ConfigError(f"missing required field: {'.'.join(path)}")
        cur = cur[key]
    return cur


def load_model_config(path) -> ModelConfig:
    """Parse + validate a model YAML into a `ModelConfig`."""
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ConfigError(f"{p}: top-level YAML must be a mapping")

    arch = _require(raw, "arch")
    precision = _require(raw, "precision")
    footprint = _require(raw, "footprint")
    pool = _require(raw, "pool")
    overheads = raw.get("overheads", {})
    fit_defaults = raw.get("fit_defaults", {})
    source = raw.get("source", {})

    return ModelConfig(
        name=_require(raw, "name"),
        num_layers=int(_require(arch, "num_layers")),
        hidden_size=int(_require(arch, "hidden_size")),
        num_attention_heads=int(_require(arch, "num_attention_heads")),
        num_key_value_heads=int(_require(arch, "num_key_value_heads")),
        head_dim=int(_require(arch, "head_dim")),
        vocab_size=int(_require(arch, "vocab_size")),
        max_context=int(_require(arch, "max_context")),
        tie_word_embeddings=bool(arch.get("tie_word_embeddings", False)),
        num_local_experts=int(arch.get("num_local_experts", 0)),
        num_experts_per_tok=int(arch.get("num_experts_per_tok", 0)),
        sliding_window=(int(arch["sliding_window"]) if arch.get("sliding_window") else None),
        quant_method=str(_require(raw, "quant", "method")) if isinstance(raw.get("quant"), dict) else "none",
        kv_dtype=str(precision.get("kv_dtype", "float16")),
        embedding_dtype=str(precision.get("embedding_dtype", "bfloat16")),
        kv_model=str(precision.get("kv_model", "full")),
        total_weight_bytes=int(_require(footprint, "total_weight_bytes")),
        framework_overhead_bytes=int(overheads.get("framework_overhead_bytes", 0)),
        activation_buffer_bytes=int(overheads.get("activation_buffer_bytes", 0)),
        target_k=int(fit_defaults.get("target_k", 1)),
        fit_max_context=int(fit_defaults.get("max_context", arch.get("max_context", 0) or 0)),
        instance_type=str(_require(pool, "instance_type")),
        gpu=str(pool.get("gpu", "")),
        gpu_vram_bytes=int(_require(pool, "gpu_vram_bytes")),
        gpu_count_per_node=int(pool.get("gpu_count_per_node", 1)),
        license=str(source.get("license", raw.get("license", "unknown"))),
        hf_repo=str(source.get("hf_repo", "")),
    )
