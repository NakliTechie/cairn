from cairn_scheduler import calcs


def test_kv_bytes_per_token_per_layer(model_cfg):
    # GQA: 2 (K+V) * 8 kv-heads * 128 head_dim * 2 bytes (bf16) = 4096 B/token/layer.
    assert calcs.kv_bytes_per_token_per_layer(model_cfg) == 2 * 8 * 128 * 2


def test_usable_vram(model_cfg):
    c = model_cfg
    # §12.1: usable = the (MEASURED) card VRAM minus the framework + activation reserves. Derive from the
    # config so this doesn't go stale when the VRAM is re-measured (it was a 24 GiB placeholder → 21.96 GiB).
    assert calcs.usable_vram_bytes(c) == c.gpu_vram_bytes - c.framework_overhead_bytes - c.activation_buffer_bytes
    # the reserves are the documented conservative design values (replace when the sglang-loaded measure lands).
    assert c.framework_overhead_bytes == 3 * 1024**3      # 3 GiB CUDA context + framework
    assert c.activation_buffer_bytes == 512 * 1024**2     # 0.5 GiB activation buffer


def test_embedding_and_lm_head(model_cfg):
    emb = calcs.embedding_bytes(model_cfg)
    assert emb == 128256 * 4096 * 2          # bf16 embedding (vocab × hidden)
    assert calcs.lm_head_bytes(model_cfg) == emb  # not tied → separate, same shape


def test_per_layer_weight_positive(model_cfg):
    plw = calcs.per_layer_weight_bytes(model_cfg)
    assert plw > 0
    # sanity: 32 layers + embedding + lm_head reconstructs ≈ total footprint.
    recon = plw * 32 + calcs.embedding_bytes(model_cfg) + calcs.lm_head_bytes(model_cfg)
    assert abs(recon - model_cfg.total_weight_bytes) < 32  # only integer-division remainder


def test_k_max_decreases_with_more_layers(model_cfg):
    ctx = 8192
    k_small_block = calcs.k_max_for_node(model_cfg, num_layers=1, context_len=ctx)
    k_big_block = calcs.k_max_for_node(model_cfg, num_layers=8, context_len=ctx)
    assert k_small_block > k_big_block >= 0  # more layers/node ⇒ more weights+KV ⇒ fewer streams
