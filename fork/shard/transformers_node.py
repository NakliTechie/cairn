"""TransformersNodeRuntime — path-C reference block runtime (no SGLang).

The cheapest rung-2 stepping stone (see sglang_node.py docstring, path C): a real-numerics
block-forward via HF transformers, CPU-runnable, to prove split-correctness + KV-replay before
the SGLang integration. Drop-in for the `NodeRuntime` contract — pass
`runtime_cls=TransformersNodeRuntime` to `adapter.build_shard_pipeline` and the SAME scheduler
drives it (sim ↔ mock ↔ this ↔ SGLang is a `runtime_cls` swap, nothing else).

It loads the full model and uses the `[start, end)` slice, **re-indexed 0-based with its own
per-seq KV** — which is what makes the split token-for-token identical to the unsplit model
(invariant #1). The memory-efficient subset-load is the SGLang path (A/B); here we only need
numerical correctness, so loading the whole model and slicing is fine (reference, not perf).
"""
from __future__ import annotations

from typing import Any, Dict

from .node import LayerRange, NodeRuntime

try:  # importable on a box without torch (structural checks); deps load lazily in load_shard
    import torch  # noqa: F401
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


class TransformersNodeRuntime(NodeRuntime):
    """One contiguous block of layers, run via HF transformers. Reference path (C)."""

    def __init__(self, model: str, layer_range: LayerRange, device: str = "cpu", dtype=None) -> None:
        super().__init__(model, layer_range, device)
        self._dtype = dtype
        self._loaded = False
        self._caches: Dict[str, Any] = {}   # seq_id -> DynamicCache for THIS block's layers

    def load_shard(self) -> None:
        if not _HAS_TORCH:
            raise RuntimeError("torch+transformers required for TransformersNodeRuntime (rung-2)")
        import torch
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_pretrained(self.model, dtype=self._dtype or torch.float32)
        m.eval()
        if self.device != "cpu":
            m.to(self.device)
        inner = m.model
        n_layers = len(inner.layers)
        s, e = self.layer_range.start, self.layer_range.end   # end EXCLUSIVE (LayerRange convention)
        self._layers = inner.layers[s:e]
        for i, lyr in enumerate(self._layers):
            lyr.self_attn.layer_idx = i                       # re-index block-local: a node holds only its slice
        self._is_embed = s == 0
        self._is_tail = e == n_layers
        self._embed = inner.embed_tokens if self._is_embed else None
        self._norm = inner.norm if self._is_tail else None
        self._lm_head = m.lm_head if self._is_tail else None
        self._inner, self._config, self._model = inner, m.config, m
        self._loaded = True

    def forward(self, hidden_states: Any, kv_meta: Dict[str, Any]) -> Any:
        import torch
        from transformers.cache_utils import DynamicCache
        from transformers.masking_utils import create_causal_mask

        if not self._loaded:
            raise RuntimeError("TransformersNodeRuntime.forward: call load_shard() first")
        seq = kv_meta["seq"]
        pos = int(kv_meta["pos"])
        cache = self._caches.setdefault(seq, DynamicCache())
        with torch.no_grad():
            # stage 0 receives token ids [1, S] and embeds; later stages receive hidden [1, S, H].
            h = self._embed(hidden_states) if self._embed is not None else hidden_states
            S = h.shape[1]
            cache_position = torch.arange(pos, pos + S, device=h.device)
            position_ids = cache_position.unsqueeze(0)
            causal_mask = create_causal_mask(
                config=self._config, inputs_embeds=h, attention_mask=None,
                past_key_values=cache, position_ids=position_ids,
            )
            pos_emb = self._inner.rotary_emb(h, position_ids=position_ids)
            for lyr in self._layers:
                h = lyr(h, attention_mask=causal_mask, position_ids=position_ids,
                        past_key_values=cache, use_cache=True, cache_position=cache_position,
                        position_embeddings=pos_emb)
            if self._is_tail:
                h = self._lm_head(self._norm(h))               # tail: norm + lm_head → logits [1, S, V]
            return h

    def free_seq(self, seq_id: str) -> None:
        """Drop a seq's KV on this block (on completion, or before a replay-rebuild)."""
        self._caches.pop(seq_id, None)

    def kv_tokens(self, seq_id: str) -> int:
        cache = self._caches.get(seq_id)
        try:
            return int(cache.get_seq_length()) if cache is not None else 0
        except Exception:
            return 0

    def heartbeat(self) -> dict:
        return {
            "loaded": self._loaded,
            "layers": [self.layer_range.start, self.layer_range.end],
            "device": self.device,
            "kv_seqs": len(self._caches),
            "alive": self._loaded,
        }
