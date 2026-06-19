"""Cairn scheduler + simulation core.

The core net-new build (spec §4). This package is built simulation-first per the
§6 cost-ladder: pure logic (model-config, the §12 calcs, the §3 fit algorithm) and a
no-GPU `BlockRuntime` mock, all headless-testable, before any GPU / AWS / Shard fork.

Public surface:
- `ModelConfig`, `load_model_config` — model-as-config (invariant #3).
- calcs — the §12 spec-time computations (VRAM, KV, K_max, real N).
- `fit` — the §3 greedy-balanced contiguous-layer placement.
- `BlockRuntime`, `MockBlockRuntime` — the per-node block seam (fork plugs in here later).
- `NodeState` — the §5.4 node state machine.
"""

from .model_config import ModelConfig, load_model_config
from .domain import NodeState, BlockAssignment, FitResult
from . import calcs
from .fit import fit, FitError

__all__ = [
    "ModelConfig",
    "load_model_config",
    "NodeState",
    "BlockAssignment",
    "FitResult",
    "calcs",
    "fit",
    "FitError",
]
