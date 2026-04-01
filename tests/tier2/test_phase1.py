import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from src.models.llama.tier2_phase1_extraction import harvest_activations, extract_transmla_matrices_for_layer

class DummyTokenizer:
    def __call__(self, texts, return_tensors="pt"):
        class Output:
            input_ids = torch.randint(0, 100, (1, 128))
        return Output()

def test_harvest_and_extract():
    config = LlamaConfig(
        vocab_size=100, # tiny dummy vocab
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128
    )
    model = LlamaForCausalLM(config)
    tokenizer = DummyTokenizer()
    
    dataset = {"text": ["Smoke test calibration text " * 50]}
    
    # We specify 1 sample of length 64. Our dummy tokenizer gives us 128 tokens total.
    all_K, all_V = harvest_activations(model, tokenizer, dataset, num_samples=1, seq_len=64)
    
    assert len(all_K) == 2
    assert len(all_V) == 2
    
    # Verify shape
    # [num_samples, seq_len, num_kv_heads * head_dim]
    # num_kv_heads = 4. head_dim = 256 / 8 = 32. 
    # width = 4 * 32 = 128
    assert all_K[0].shape == (1, 64, 128)
    
    layer_idx = 0
    matrices = extract_transmla_matrices_for_layer(
        layer_idx=layer_idx,
        K_layer=all_K[layer_idx],
        V_layer=all_V[layer_idx],
        num_kv_heads=config.num_key_value_heads,
        head_dim=32,
        target_rank=16, # Compress down to 16
        rope_retained_dim=8 # Keep 8 dimensions for RoPE out of half_dim=16
    )
    
    assert "U_l" in matrices
    assert "R_KV" in matrices
    assert "alpha" in matrices
    
    # U_l acts on half_dim = 16
    assert matrices["U_l"].shape == (16, 16)
    
    # R_KV projects NOPE Keys and Values
    # NOPE Keys = 32 - 8 = 24 dimensions
    # Values = 32 dimensions
    # total input to PCA = (24 + 32) * 4 = 224
    assert matrices["R_KV"].shape == (224, 16)
