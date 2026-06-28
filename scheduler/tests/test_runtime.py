from cairn_scheduler import fit
from cairn_scheduler.runtime import (
    MockBlockRuntime,
    build_mock_pipeline,
    chain_forward,
    sample,
)


def _decode(stages, stream_id, prompt_hidden, steps, vocab):
    """Greedy decode loop in sim: traverse the pipeline, sample, feed the token back."""
    tokens = []
    h = prompt_hidden
    for pos in range(steps):
        out = chain_forward(stages, stream_id, h, pos)
        tok = sample(out, vocab)
        tokens.append(tok)
        h = tok  # sampled token becomes the next step's input (autoregressive)
    return tokens


def test_forward_grows_kv():
    stage = MockBlockRuntime(0, 0, 3)
    assert stage.kv_len("a") == 0
    stage.forward("a", 111, 0)
    stage.forward("a", 222, 1)
    assert stage.kv_len("a") == 2
    stage.free_stream("a")
    assert stage.kv_len("a") == 0 and not stage.has_stream("a")


def test_cross_stream_kv_isolation(model_cfg):
    """Spec §4.4: a token of stream A must never read stream B's KV. Interleaving B
    must not change A's output stream."""
    r = fit(model_cfg, target_k=4, context_len=2048)
    vocab = model_cfg.vocab_size

    solo = build_mock_pipeline(r)
    a_solo = _decode(solo, "A", prompt_hidden=7, steps=6, vocab=vocab)

    interleaved = build_mock_pipeline(r)
    a_inter = []
    hb = 99
    ha = 7
    for pos in range(6):
        oa = chain_forward(interleaved, "A", ha, pos)
        ta = sample(oa, vocab)
        a_inter.append(ta)
        ha = ta
        # interleave a B-token between every A-token through the SAME stages
        ob = chain_forward(interleaved, "B", hb, pos)
        hb = sample(ob, vocab)

    assert a_inter == a_solo  # B's presence left no trace on A


def test_split_equals_single_reference(model_cfg):
    """Spec §9 v1.0 correctness gate (in sim): the split pipeline is token-for-token
    identical to a single-node reference holding all layers."""
    r = fit(model_cfg, target_k=16, context_len=32768)  # force a real multi-stage split
    vocab = model_cfg.vocab_size

    split = build_mock_pipeline(r)
    reference = [MockBlockRuntime(0, 0, model_cfg.num_layers - 1)]

    ref_tokens = _decode(reference, "S", prompt_hidden=42, steps=12, vocab=vocab)
    split_tokens = _decode(split, "S", prompt_hidden=42, steps=12, vocab=vocab)
    assert split_tokens == ref_tokens
    assert r.n >= 2  # the gate is only meaningful if it actually split into >1 stage


def test_replay_rebuilds_kv(model_cfg):
    """Spec §5.3: KV is derived; on a drop the block's cache is rebuilt by re-prefilling
    over the durable token-history. Re-running the history must reproduce identical output."""
    stage = MockBlockRuntime(2, 8, 15)  # a middle block
    history = [13, 21, 34, 55, 89]      # the stream's hidden inputs to this stage so far

    for pos, h in enumerate(history):
        stage.forward("S", h, pos)
    before = stage.forward("S", 144, len(history))  # the next token's output, pre-crash

    # node dies → its KV is gone; rebuild by replaying the history on a fresh runtime.
    rebuilt = MockBlockRuntime(2, 8, 15)
    for pos, h in enumerate(history):
        rebuilt.forward("S", h, pos)
    after = rebuilt.forward("S", 144, len(history))

    assert after == before  # replay reconstructed the exact cache state
