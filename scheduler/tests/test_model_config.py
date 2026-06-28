import pytest

from cairn_scheduler.model_config import ConfigError, dtype_bytes, load_model_config


def test_load_llama(model_cfg):
    c = model_cfg
    assert c.name == "llama-3.1-8b"
    assert c.num_layers == 32
    assert c.num_key_value_heads == 8       # GQA — the KV-cache driver
    assert c.head_dim == 128
    assert c.vocab_size == 128256
    assert not c.is_moe and c.num_local_experts == 0   # dense
    assert c.license == "Llama-3.1-community"
    assert c.gpu_vram_bytes == 25769803776   # 24 GiB — L4 nominal
    assert c.tie_word_embeddings is False


def test_load_qwen_skeleton(configs_dir):
    c = load_model_config(configs_dir / "qwen3.5-397b.yaml")
    assert c.name.startswith("qwen3.5")
    assert c.is_moe
    assert c.gpu == "L40S"
    assert c.quant_method == "fp8"


def test_missing_required_field_raises(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: x\narch: {num_layers: 4}\n")  # missing precision/footprint/pool
    with pytest.raises(ConfigError):
        load_model_config(bad)


def test_zero_layers_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "name: x\n"
        "arch: {num_layers: 0, hidden_size: 8, num_attention_heads: 1, "
        "num_key_value_heads: 1, head_dim: 8, vocab_size: 10, max_context: 16}\n"
        "precision: {kv_dtype: float16, embedding_dtype: bfloat16, kv_model: full}\n"
        "footprint: {total_weight_bytes: 1000}\n"
        "pool: {instance_type: x, gpu_vram_bytes: 1000}\n"
    )
    with pytest.raises(ConfigError):
        load_model_config(bad)


def test_dtype_bytes():
    assert dtype_bytes("float16") == 2
    assert dtype_bytes("bf16") == 2
    assert dtype_bytes("fp8") == 1
    assert dtype_bytes("mxfp4") == 0.5
    assert dtype_bytes("int4") == 0.5
    with pytest.raises(ConfigError):
        dtype_bytes("not-a-dtype")
