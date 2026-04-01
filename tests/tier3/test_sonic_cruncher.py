import pytest
import torch
import torch.nn as nn
from src.compression.sonic import SonicCruncher
from src.tiers.tier3_sonic import Tier3Sonic

def test_sonic_cruncher_forward_4d():
    """ Tests forward pass of SonicCruncher with 4D tensors (batch, heads, seq, dim) """
    batch, num_heads, seq_len, dim = 2, 4, 15, 64
    max_k = 4
    cruncher = SonicCruncher(dim=dim, max_k=max_k)
    
    # Mock N-length latents from Tier 2
    c_kv = torch.randn(batch, num_heads, seq_len, dim)
    
    # Dynamic budget K=2
    k = 2
    c_nexus = cruncher(c_kv, k=k)
    
    assert c_nexus.shape == (batch, num_heads, k, dim)

def test_sonic_cruncher_forward_3d():
    """ Tests forward pass of SonicCruncher with 3D tensors (batch, seq, dim) """
    batch, seq_len, dim = 2, 10, 64
    max_k = 4
    cruncher = SonicCruncher(dim=dim, max_k=max_k)
    
    c_kv = torch.randn(batch, seq_len, dim)
    
    k = 4
    c_nexus = cruncher(c_kv, k=k)
    
    assert c_nexus.shape == (batch, k, dim)

def test_sonic_cruncher_gradients():
    """ Tests that gradients correctly flow back to nexus_base and projections """
    dim = 32
    cruncher = SonicCruncher(dim=dim, max_k=2)
    
    batch, num_heads, seq_len = 1, 2, 8
    c_kv = torch.randn(batch, num_heads, seq_len, dim, requires_grad=True)
    
    c_nexus = cruncher(c_kv, k=1)
    
    loss = c_nexus.sum()
    loss.backward()
    
    # Ensure gradients are populated
    assert cruncher.nexus_base.grad is not None
    assert cruncher.q_proj.weight.grad is not None
    assert cruncher.k_proj.weight.grad is not None
    assert cruncher.v_proj.weight.grad is not None
    assert cruncher.o_proj.weight.grad is not None

def test_tier3_sonic_append():
    """ Tests appending to Tier3Sonic """
    tier3 = Tier3Sonic(capacity=50, layer_idx=0)
    
    batch, num_heads, k, dim = 2, 4, 2, 64
    
    # Push first block
    c_nexus_1 = torch.randn(batch, num_heads, k, dim)
    k_rope_1 = torch.randn(batch, num_heads, k, dim)
    
    evicted_c, evicted_k = tier3.push(c_nexus_1, k_rope_1)
    
    assert evicted_c is None
    assert evicted_k is None
    assert tier3.seq_len == 2
    
    c_nexus_cached, k_rope_cached = tier3.get_cache()
    assert torch.equal(c_nexus_cached, c_nexus_1)
    assert torch.equal(k_rope_cached, k_rope_1)
    
    # Push second block
    c_nexus_2 = torch.randn(batch, num_heads, k, dim)
    k_rope_2 = torch.randn(batch, num_heads, k, dim)
    tier3.push(c_nexus_2, k_rope_2)
    
    assert tier3.seq_len == 4
    c_nexus_cached, k_rope_cached = tier3.get_cache()
    assert c_nexus_cached.shape[2] == 4
    assert torch.equal(c_nexus_cached[:, :, 2:4, :], c_nexus_2)

def test_tier3_sonic_eviction():
    """ Tests eviction from Tier3Sonic when capacity is full """
    tier3 = Tier3Sonic(capacity=5, layer_idx=0)
    
    batch, num_heads, k, dim = 1, 2, 4, 32
    
    # Push 4 tokens (under capacity of 5)
    c_nexus_1 = torch.randn(batch, num_heads, 4, dim)
    k_rope_1 = torch.randn(batch, num_heads, 4, dim)
    
    evicted_c, evicted_k = tier3.push(c_nexus_1, k_rope_1)
    assert evicted_c is None
    assert tier3.seq_len == 4
    
    # Push 3 tokens (total 7, exceeds capacity by 2)
    c_nexus_2 = torch.randn(batch, num_heads, 3, dim)
    k_rope_2 = torch.randn(batch, num_heads, 3, dim)
    
    evicted_c, evicted_k = tier3.push(c_nexus_2, k_rope_2)
    
    # Should evict the first 2 tokens
    assert evicted_c is not None
    assert evicted_c.shape == (batch, num_heads, 2, dim)
    assert torch.equal(evicted_c, c_nexus_1[:, :, :2, :])
    
    assert evicted_k is not None
    assert torch.equal(evicted_k, k_rope_1[:, :, :2, :])
    
    assert tier3.seq_len == 5
    
    c_nexus_cached, _ = tier3.get_cache()
    # Cache should contain exactly the last 2 from c_nexus_1 and all 3 from c_nexus_2
    assert torch.equal(c_nexus_cached[:, :, :2, :], c_nexus_1[:, :, 2:, :])
    assert torch.equal(c_nexus_cached[:, :, 2:, :], c_nexus_2)
