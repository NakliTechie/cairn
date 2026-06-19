from cairn_scheduler import calcs


def test_kv_bytes_per_token_per_layer(gpt_oss_cfg):
    # GQA: 2 (K+V) * 8 kv-heads * 64 head_dim * 2 bytes (fp16) = 2048 B/token/layer.
    assert calcs.kv_bytes_per_token_per_layer(gpt_oss_cfg) == 2 * 8 * 64 * 2


def test_usable_vram(gpt_oss_cfg):
    # card 24 GiB − 3 GiB framework − 0.5 GiB activation.
    expected = 24 * 1024**3 - 3 * 1024**3 - 512 * 1024**2
    assert calcs.usable_vram_bytes(gpt_oss_cfg) == expected


def test_embedding_and_lm_head(gpt_oss_cfg):
    emb = calcs.embedding_bytes(gpt_oss_cfg)
    assert emb == 201088 * 2880 * 2          # bf16 embedding
    assert calcs.lm_head_bytes(gpt_oss_cfg) == emb  # not tied → separate, same shape


def test_per_layer_weight_positive(gpt_oss_cfg):
    plw = calcs.per_layer_weight_bytes(gpt_oss_cfg)
    assert plw > 0
    # sanity: 36 layers + embedding + lm_head reconstructs ≈ total footprint.
    recon = plw * 36 + calcs.embedding_bytes(gpt_oss_cfg) + calcs.lm_head_bytes(gpt_oss_cfg)
    assert abs(recon - gpt_oss_cfg.total_weight_bytes) < 36  # only integer-division remainder


def test_k_max_decreases_with_more_layers(gpt_oss_cfg):
    ctx = 8192
    k_small_block = calcs.k_max_for_node(gpt_oss_cfg, num_layers=1, context_len=ctx)
    k_big_block = calcs.k_max_for_node(gpt_oss_cfg, num_layers=8, context_len=ctx)
    assert k_small_block > k_big_block >= 0  # more layers/node ⇒ more weights+KV ⇒ fewer streams
