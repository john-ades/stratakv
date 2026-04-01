import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import pytest

from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
from src.models.llama.modeling_llama import patch_llama_for_strata

@pytest.fixture
def dummy_llama():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64, # head_dim = hidden_size / num_heads = 64 / 4 = 16
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4, # head_dim = 16
        num_key_value_heads=2, # GQA
        max_position_embeddings=1024
    )
    model = LlamaForCausalLM(config)
    return model

@pytest.fixture
def strata_config():
    return StrataKVConfig(
        num_kv_heads=2,
        head_dim=16,
        
        # Tiering
        enable_tier0=True,
        tier0_size=4,
        
        enable_tier1=True,
        tier1_size=8,
        
        enable_tier2=True,
        tier2_size=100,
        
        # Compression
        transmla_rope_dim=8,  # out of 16 half_dim is 8, wait, RoPE dimension should be <= head_dim. half_dim=8.
        transmla_target_rank=12,
        transmla_matrices_path=None
    )

def test_hybrid_attention_forward_and_backward(dummy_llama, strata_config):
    # 1. Patch the model
    patch_llama_for_strata(dummy_llama, strata_config)
    
    # Check that absorbers were added
    for name, module in dummy_llama.named_modules():
        if module.__class__.__name__ == "LlamaAttention":
            assert hasattr(module, "strata_absorber")
            assert module.strata_absorber.target_rank == 12
            
    # Modify absorber weights to require grad for phase 5
    for name, param in dummy_llama.named_parameters():
        if "strata_absorber" in name and ("W_UK" in name or "W_UV" in name):
            param.requires_grad = True

    # Initialize TransMLA mock parameters to non-zero so gradients are not identically 0
    with torch.no_grad():
        for layer in dummy_llama.model.layers:
            layer.self_attn.strata_absorber.W_UK.normal_()
            layer.self_attn.strata_absorber.W_UV.normal_()
            layer.self_attn.strata_cruncher.R_KV.normal_()
    
    # 2. Simulate an autoregressive generation where cache gets filled
    # First pass: Generate initial tokens to push things into Tier 2.
    # Total capacity before Tier 2 = 4 + 8 = 12. 
    # Let's do a prompt of 20 tokens to force 8 tokens into Tier 2.
    
    bsz = 1
    seq_len = 20
    input_ids = torch.randint(0, 100, (bsz, seq_len))
    
    cache = StrataKVCache(strata_config)
    
    # Enable grad
    dummy_llama.train()
    
    # Forward pass
    outputs = dummy_llama(
        input_ids=input_ids,
        past_key_values=cache,
        use_cache=True,
        output_attentions=True
    )
    
    logits = outputs.logits
    # The output should just be normal tensors
    assert logits.shape == (bsz, seq_len, 100)
    
    # 3. Verify that Tier 2 cache actually received tokens
    # Layer 0 cache details
    t0_k, _ = cache._tier0_sinks[0].get_cache()
    t1_k, _ = cache._tier1_recents[0].get_cache()
    c_kv, k_rope = cache._tier2_latents[0].get_cache()
    
    assert c_kv is not None, "Tier 2 cache should have tokens"
    assert c_kv.shape[1] == 8, f"Expected 8 tokens in Tier 2, got {c_kv.shape[1]}"
    assert k_rope.shape[2] == 8, f"Expected 8 tokens in K rope, got {k_rope.shape[2]}"
    
    # 4. Try backward pass to ensure gradients flow into W_UK and W_UV
    loss = logits.sum()
    loss.backward()
    
    # Check W_UK and W_UV gradients
    absorber = dummy_llama.model.layers[0].self_attn.strata_absorber
    assert absorber.W_UK.grad is not None, "W_UK did not receive gradients"
    assert absorber.W_UV.grad is not None, "W_UV did not receive gradients"
    
    # The loss backpropagated through `attn_outputs = out_t1 + out_t2`
    
def test_mask_alignment_during_hybrid_attention(dummy_llama, strata_config):
    # Ensure that generating the next token works perfectly
    patch_llama_for_strata(dummy_llama, strata_config)
    cache = StrataKVCache(strata_config)
    
    dummy_llama.eval()
    
    with torch.no_grad():
        # Pass 1: 15 tokens
        outputs = dummy_llama(input_ids=torch.randint(0, 100, (1, 15)), past_key_values=cache, use_cache=True)
        
        # Pass 2: 1 token
        next_id = torch.randint(0, 100, (1, 1))
        outputs2 = dummy_llama(input_ids=next_id, past_key_values=cache, use_cache=True)
        
        # If it doesn't crash from an attention mask mismatch, we succeed!
        assert outputs2.logits.shape == (1, 1, 100)
