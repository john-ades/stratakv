import torch
import pytest
from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
from src.tiers.tier2_latent import Tier2Latent

def test_tier2_latent_cache():
    # Test pushing vectors into Tier 2 and capacity evictions
    tier2 = Tier2Latent(capacity=100, layer_idx=0)
    
    # Simulate c_kv: [batch, seq_len, r_kv]
    # Simulate k_rope: [batch, seq_len, rope_dim]
    batch, seq_len, r_kv, rope_dim = 2, 60, 64, 16
    c_kv1 = torch.randn(batch, seq_len, r_kv)
    k_rope1 = torch.randn(batch, seq_len, rope_dim)
    
    evicted_c, evicted_k = tier2.push(c_kv1, k_rope1)
    
    # Since capacity is 100, pushing 60 should not evict
    assert evicted_c is None
    assert evicted_k is None
    
    # Query length should be 60
    assert tier2.seq_len == 60
    
    # Push another 60 to exceed capacity
    c_kv2 = torch.randn(batch, seq_len, r_kv)
    k_rope2 = torch.randn(batch, seq_len, rope_dim)
    
    evicted_c2, evicted_k2 = tier2.push(c_kv2, k_rope2)
    
    # Now it holds 120 tokens, capacity is 100, so 20 should be evicted.
    assert evicted_c2 is not None
    assert evicted_k2 is not None
    assert evicted_c2.shape == (batch, 20, r_kv)
    assert evicted_k2.shape == (batch, 20, rope_dim)
    
    assert tier2.seq_len == 100
    
    # Internal stored shapes should be 100
    stored_c, stored_k = tier2.get_cache()
    assert stored_c is not None
    assert stored_c.shape == (batch, 100, r_kv)
    assert stored_k.shape == (batch, 100, rope_dim)

def test_stratakvcache_tier2_config():
    # Enable tier 2
    config = StrataKVConfig(enable_tier0=True, tier0_size=4, enable_tier1=True, tier1_size=10, enable_tier2=True, tier2_size=100)
    
    # Instantiate manager
    cache_manager = StrataKVCache(config)
    
    # Before update, sequence length should be 0
    assert cache_manager.get_seq_length(layer_idx=0) == 0
    
    # Max length should be 4 + 10 + 100 = 114
    assert cache_manager.get_max_length() == 114
    
    # Provide dummy kv state to initialize layer 0 tiers
    batch, num_heads, seq_len, head_dim = 1, 2, 8, 32
    k_state = torch.randn(batch, num_heads, seq_len, head_dim)
    v_state = torch.randn(batch, num_heads, seq_len, head_dim)
    
    cache_manager.update(k_state, v_state, layer_idx=0)
    
    # Verify tier2 is initialized
    assert len(cache_manager._tier2_latents) > 0
    assert cache_manager._tier2_latents[0] is not None
    assert cache_manager._tier2_latents[0].capacity == 100
    
    # We pushed 8 tokens. 4 go to Tier0, 4 go to Tier1. 0 in Tier2.
    assert cache_manager.get_seq_length(layer_idx=0) == 8
    
    # Since Phase 3 (Cruncher) isn't implemented, tier 2 internal seq_len should still be 0 
    assert cache_manager._tier2_latents[0].seq_len == 0
