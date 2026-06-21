"""SGLang split-correctness on REAL kernels (GPU) — the on-box executable target for the rung-2 forward.

The path-C transformers oracle (test_ref_split.py) already banked the split MATH on CPU. This is the
SGLang counterpart on a real GPU: the SAME two invariants, driven through
`build_shard_pipeline(runtime_cls=SglangNodeRuntime)` so the SGLang kernels / paging / quant are exercised.

  - split-correctness (inv #1): SGLang split by contiguous layers (each block re-indexed, own paged KV)
    == SGLang UNSPLIT (one stage, all layers), token-for-token, at any cut points. The reference is
    SGLang-unsplit, NOT transformers — both sides run identical kernels, so any divergence is purely the
    split / stitch / KV logic (exactly what rung-2 must prove), free of SGLang-vs-HF numeric noise.
  - replay-rebuild (inv #2): a FRESH pipeline given only the TOKEN HISTORY rebuilds the paged KV and
    resumes greedy decode bit-identically (recovery = replay; KV is derived, never checkpointed).

RUN ON THE BOX. This FAILS until SglangNodeRuntime.load_shard/forward are implemented — that is the
point: it is the spec. Iterate the forward, run this, until green (no test scaffolding on metered GPU).
GPU + sglang gated (skips otherwise).

  CAIRN_TEST_MODEL   HF id of a small model to split (default below). Must be stage-able on the box
                     (HF_TOKEN if gated). The exact tokens/model don't affect split==unsplit; pick small.
  CAIRN_TEST_DEVICE  default cuda:0 — all stages share one GPU here (the split-CORRECTNESS gate, Step 3).
                     The true 2-GPU + wire run is Step 4 (RUNBOOK-rung2).

⚠ quant: build_shard_pipeline constructs SglangNodeRuntime with its constructor-default quant. For an
unquantized small test model the forward must load it un-quantized (derive quant from the model/Cairn
config — don't force mxfp4). Threading quant through build_shard_pipeline is a box-side follow-on; both
pipelines here are built identically, so the split==unsplit comparison stays valid regardless.
"""
import os
import pathlib
import sys
from types import SimpleNamespace

import pytest

_FORK = pathlib.Path(__file__).resolve().parents[1]
if str(_FORK) not in sys.path:
    sys.path.insert(0, str(_FORK))

torch = pytest.importorskip("torch")
pytest.importorskip("sglang")
if not torch.cuda.is_available():  # pragma: no cover - GPU-only
    pytest.skip("SGLang block-forward is GPU-only (rung-2)", allow_module_level=True)

from adapter import build_shard_pipeline  # noqa: E402
from shard.sglang_node import SglangNodeRuntime  # noqa: E402

MODEL = os.environ.get("CAIRN_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
DEVICE = os.environ.get("CAIRN_TEST_DEVICE", "cuda:0")
PROMPT = [1, 5, 9, 3, 7, 2, 11, 4]   # arbitrary valid token ids — split==unsplit is token-agnostic
NEW = 12


def _num_layers(model):
    """Read depth from config.json without loading weights (transformers is on the box via sglang)."""
    from transformers import AutoConfig
    return AutoConfig.from_pretrained(model).num_hidden_layers


def _fake_fit(num_layers, cuts):
    """A FitResult-shaped split: contiguous, inclusive layer_start/layer_end (Cairn convention).
    cuts=[] → a single stage holding all layers (the UNSPLIT reference)."""
    b = [0, *cuts, num_layers]
    return SimpleNamespace(
        assignments=[SimpleNamespace(stage=i, layer_start=b[i], layer_end=b[i + 1] - 1)
                     for i in range(len(b) - 1)],
        n=len(b) - 1,
    )


def _make_pipe(num_layers, cuts):
    pipe = build_shard_pipeline(_fake_fit(num_layers, cuts), MODEL, device=DEVICE,
                                runtime_cls=SglangNodeRuntime)
    for blk in pipe:
        blk.load()
    return pipe


def _drive(pipe, init_ids, steps):
    """Greedy-decode `steps` tokens (the SAME driver as the path-C oracle): the first forward ingests
    init_ids (prefill / replay), each next forward is one token. Stage 0 embeds ids; the tail returns
    logits [1, S, V]."""
    cur, out, pos = torch.tensor([list(init_ids)], device=DEVICE), [], 0
    with torch.no_grad():
        for _ in range(steps):
            seq_len, h = cur.shape[1], cur
            for blk in pipe:
                h = blk.forward("s0", h, pos)
            nxt = h[:, -1].argmax(-1)
            out.append(int(nxt))
            cur = nxt.unsqueeze(0)
            pos += seq_len
    return out


@pytest.fixture(scope="module")
def num_layers():
    L = _num_layers(MODEL)
    if L < 3:  # pragma: no cover - real test models have tens of layers
        pytest.skip(f"{MODEL} has {L} layers; need >=3 to exercise multi-cut splits")
    return L


def _cuts(L, which):
    return {"2-block": [L // 2], "3-block": [L // 3, 2 * L // 3]}[which]


@pytest.mark.parametrize("which", ["2-block", "3-block"])
def test_split_equals_unsplit(num_layers, which):
    """inv #1: SGLang split == SGLang unsplit (one stage), token-for-token, at any cut points."""
    ref = _drive(_make_pipe(num_layers, []), PROMPT, NEW)          # unsplit: no cuts → single stage
    assert _drive(_make_pipe(num_layers, _cuts(num_layers, which)), PROMPT, NEW) == ref


def test_replay_rebuild_resumes_uncorrupted(num_layers):
    """inv #2: a node is lost at token K; a fresh pipeline replays the TOKEN HISTORY to rebuild the paged
    KV and must resume bit-identically — proving KV is derivable from tokens (recovery = replay)."""
    cuts, kill_at = _cuts(num_layers, "3-block"), 6
    full = _drive(_make_pipe(num_layers, cuts), PROMPT, NEW)        # the uninterrupted run
    history = PROMPT + full[:kill_at]                              # the token-history "truth"
    resumed = _drive(_make_pipe(num_layers, cuts), history, NEW - kill_at)  # fresh KV, replay → resume
    assert resumed == full[kill_at:]
