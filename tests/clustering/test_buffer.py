import torch
import pytest
import numpy as np
from src.core.config import StrataKVConfig
from src.clustering.buffer import AbitClusterBuffer

def test_tensor_to_semantic_vector():
    config = StrataKVConfig()
    buffer = AbitClusterBuffer(config)

    # 3D Tensor: [batch=1, seq=1, dim=128]
    dim = 128
    c_kv_3d = torch.ones(1, 1, dim) * 0.5
    vec_1 = buffer._tensor_to_semantic_vector(c_kv_3d)
    assert vec_1.shape == (1, dim)

    # 4D Tensor: [batch=1, num_heads=4, seq=1, dim=32] => meaning total dim 128
    c_kv_4d = torch.ones(1, 4, 1, 32) * 0.5
    vec_2 = buffer._tensor_to_semantic_vector(c_kv_4d)
    assert vec_2.shape == (1, 32)

def test_push_and_seal():
    config = StrataKVConfig(
        abit_window_size=3,
        abit_min_split_tokens=2,
        abit_max_split_tokens=4,
        abit_initial_threshold=0.5
    )
    buffer = AbitClusterBuffer(config)

    dim = 128
    # We will simulate tokens that should trigger a split.
    # First token: completely zeros
    c_kv_1 = torch.zeros(1, 1, 1, dim)
    k_rope_1 = torch.zeros(1, 1, 1, dim)

    # Push first token -> buffer size 1, no clusters
    clusters = buffer.push(c_kv_1, k_rope_1)
    assert len(clusters) == 0
    assert len(buffer._c_kv_buffer[0]) == 1

    # Second token: completely zeros -> semantic sim high, no split even if total >= min
    c_kv_2 = torch.zeros(1, 1, 1, dim)
    k_rope_2 = torch.zeros(1, 1, 1, dim)
    clusters = buffer.push(c_kv_2, k_rope_2)
    assert len(clusters) == 0
    assert len(buffer._c_kv_buffer[0]) == 2

    # Third token: orthogonal (ones) -> trigges a split, pushing the preceding 2 tokens
    c_kv_3 = torch.ones(1, 1, 1, dim)
    k_rope_3 = torch.ones(1, 1, 1, dim)
    clusters = buffer.push(c_kv_3, k_rope_3)
    
    assert len(clusters) == 1
    sealed = clusters[0]
    
    assert sealed.segment.total_tokens == 2
    assert sealed.c_kv.shape == (1, 1, 2, dim) # 4D tensor checks out
    assert torch.allclose(sealed.c_kv, torch.zeros(1, 1, 2, dim))
    
    # We pushed 3 total, 2 were sealed, 1 remains in buffer
    assert len(buffer._c_kv_buffer[0]) == 1
    # Check that it's the 3rd token
    assert torch.allclose(buffer._c_kv_buffer[0][0], c_kv_3)
    
    # Flush the rest
    leftover = buffer.flush()
    assert len(leftover) == 1
    assert leftover[0].segment.total_tokens == 1
    assert torch.allclose(leftover[0].c_kv, c_kv_3)
    
    assert len(buffer._c_kv_buffer[0]) == 0


def test_push_chunked():
    config = StrataKVConfig(
        abit_min_split_tokens=2,
        abit_max_split_tokens=3,
        abit_initial_threshold=-1.0 # Force split by max tokens
    )
    buffer = AbitClusterBuffer(config)

    dim = 64
    # Chunk of 5 tokens
    c_kv = torch.randn(1, 5, dim)
    k_rope = torch.randn(1, 5, dim)

    clusters = buffer.push(c_kv, k_rope)

    # It should hit max_split_tokens = 3 at the 4th token in the chunk
    assert len(clusters) == 1
    sealed = clusters[0]
    assert sealed.segment.total_tokens == 3
    assert sealed.c_kv.shape == (1, 3, dim) # seq dim is 1 for 3D
    assert torch.allclose(sealed.c_kv, c_kv[:, :3, :])

    # The remaining 2 tokens of the chunk should be in the buffer
    assert len(buffer._c_kv_buffer[0]) == 2
    assert torch.allclose(buffer._c_kv_buffer[0][0], c_kv[:, 3:4, :])
    assert torch.allclose(buffer._c_kv_buffer[0][1], c_kv[:, 4:5, :])
    
    # Flush remaining
    final = buffer.flush()
    assert len(final) == 1
    assert final[0].segment.total_tokens == 2
    assert torch.allclose(final[0].c_kv, c_kv[:, 3:, :])

def test_cache_manager_integration():
    from src.cache_manager import StrataKVCache
    
    config = StrataKVConfig(
        enable_tier0=False,
        enable_tier1=False,
        enable_tier2=True,
        enable_tier3=True
    )
    cache = StrataKVCache(config)
    cache._ensure_initialized(layer_idx=0)
    
    assert len(cache._tier3_buffers) == 1
    assert cache._tier3_buffers[0] is not None
    assert isinstance(cache._tier3_buffers[0], AbitClusterBuffer)
