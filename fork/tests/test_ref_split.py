"""Path-C split-correctness on REAL numerics (transformers), CPU, through build_shard_pipeline.

Proves invariant #1 beyond the integer MockBlockRuntime: a model split by contiguous layers —
each block re-indexed with its own per-seq KV — produces token-for-token identical greedy decode
to the unsplit model, at any cut points. The SAME `build_shard_pipeline` seam the fleet uses, with
`runtime_cls=TransformersNodeRuntime`. torch-gated (skipped without torch+transformers)."""
import pathlib
import sys
from types import SimpleNamespace

import pytest

_FORK = pathlib.Path(__file__).resolve().parents[1]
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from adapter import build_shard_pipeline  # noqa: E402
from shard.transformers_node import TransformersNodeRuntime  # noqa: E402

PROMPT = [1, 5, 9, 3, 7, 2]
NEW = 16


def _build_tiny(dest):
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=48, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128)
    LlamaForCausalLM(cfg).eval().save_pretrained(dest)
    return cfg.num_hidden_layers


def _fake_fit(num_layers, cuts):
    """A FitResult-shaped split: contiguous, inclusive layer_start/layer_end (Cairn convention)."""
    b = [0, *cuts, num_layers]
    return SimpleNamespace(
        assignments=[SimpleNamespace(stage=i, layer_start=b[i], layer_end=b[i + 1] - 1)
                     for i in range(len(b) - 1)],
        n=len(b) - 1,
    )


def _ref_greedy(path):
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
    cache = DynamicCache()
    cur, out = torch.tensor([PROMPT]), []
    with torch.no_grad():
        for _ in range(NEW):
            nxt = m(cur, past_key_values=cache, use_cache=True).logits[:, -1].argmax(-1)
            out.append(int(nxt))
            cur = nxt.unsqueeze(0)
    return out


def _split_greedy(path, num_layers, cuts):
    pipe = build_shard_pipeline(_fake_fit(num_layers, cuts), path, device="cpu",
                                runtime_cls=TransformersNodeRuntime)
    for blk in pipe:
        blk.load()
    cur, out, pos = torch.tensor([PROMPT]), [], 0
    with torch.no_grad():
        for _ in range(NEW):
            seq_len, h = cur.shape[1], cur
            for blk in pipe:                    # stage 0 embeds ids; hidden handed off; tail → logits
                h = blk.forward("s0", h, pos)
            nxt = h[:, -1].argmax(-1)
            out.append(int(nxt))
            cur = nxt.unsqueeze(0)
            pos += seq_len
    return out


@pytest.mark.parametrize("cuts", [[3], [2, 4], [1, 2, 3, 4, 5]],
                         ids=["2-block", "3-block", "6-block"])
def test_split_equals_single_ref(tmp_path, cuts):
    path = str(tmp_path / "tiny")
    num_layers = _build_tiny(path)
    assert _split_greedy(path, num_layers, cuts) == _ref_greedy(path)
