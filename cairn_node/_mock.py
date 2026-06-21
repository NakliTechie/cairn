"""MockNodeRuntime — a CPU/torch stand-in for SglangNodeRuntime, to validate the multi-process pipeline
plumbing (serve loop + wire + driver) WITHOUT a GPU. Its per-layer transform is deterministic and
COMPOSES, so split (layers run in stages) == unsplit (all layers) by construction — the exact property
the real SGLang test checks, exercised here on the socket/wire path for free.

Stateless (no KV), so it does NOT test replay-rebuild — that needs the real sglang paged KV on a GPU.
`model` is the string "mock:<num_layers>"."""
from __future__ import annotations

from typing import Any, Dict

from shard.node import NodeRuntime


class MockNodeRuntime(NodeRuntime):
    H = 16  # hidden width == mock vocab (the tail returns hidden AS logits)

    def __init__(self, model: str, layer_range, device: str = "cpu", quant=None) -> None:
        super().__init__(model, layer_range, device)
        self._n = int(str(model).split(":")[-1])      # "mock:<num_layers>"
        self._s, self._e = layer_range.start, layer_range.end
        self._is_embed = self._s == 0
        self._is_tail = self._e == self._n

    def load_shard(self) -> None:
        pass

    def forward(self, hidden_states: Any, kv_meta: Dict[str, Any]) -> Any:
        import torch

        x = hidden_states
        if self._is_embed:                            # [1, S] ids -> [S, H] one-hot at (id % H)
            s_len = x.shape[1]
            h = torch.zeros(s_len, self.H)
            for i in range(s_len):
                h[i, int(x[0, i]) % self.H] = 1.0
        else:                                         # [1, S, H] hidden -> flat [S, H]
            h = x.reshape(-1, self.H)
        for i in range(self._s, self._e):             # composes → split == unsplit
            h = h + float(i + 1)
        return h.reshape(1, -1, self.H)               # tail: hidden AS logits; mid: hand-off

    def heartbeat(self) -> dict:
        return {"loaded": True, "alive": True, "layers": [self._s, self._e]}
