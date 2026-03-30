import torch
import pytest
from src.compression.transmla import TransMLACruncher

def test_transmla_cruncher_shapes():
    batch_size = 2
    num_kv_heads = 4
    seq_len = 10
    head_dim = 128
    rope_retained_dim = 64
    target_rank = 32
    layer_idx = 0
    
    # Init cruncher
    cruncher = TransMLACruncher(
        layer_idx=layer_idx,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        rope_retained_dim=rope_retained_dim,
        target_rank=target_rank
    )
    
    # Dummy Tier 1 evicted tokens
    K_evicted = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    V_evicted = torch.randn(batch_size, num_kv_heads, seq_len, head_dim)
    
    # Crunch!
    C_kv, K_rope = cruncher(K_evicted, V_evicted)
    
    # Assert shapes
    assert C_kv.shape == (batch_size, seq_len, target_rank)
    assert K_rope.shape == (batch_size, num_kv_heads, seq_len, rope_retained_dim)
    
    # Verify dtype preservation if we change it
    cruncher = cruncher.to(dtype=torch.float16)
    K_evicted_fp16 = K_evicted.to(torch.float16)
    V_evicted_fp16 = V_evicted.to(torch.float16)
    
    C_kv_fp16, K_rope_fp16 = cruncher(K_evicted_fp16, V_evicted_fp16)
    assert C_kv_fp16.dtype == torch.float16
    assert K_rope_fp16.dtype == torch.float16
