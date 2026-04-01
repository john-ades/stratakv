import pytest
import torch
import numpy as np
from src.clustering.abit import ClusterSegment
from src.clustering.buffer import SealedCluster

@pytest.fixture
def mock_segment():
    # A dummy ClusterSegment with a known medoid_idx of 5
    # Let's say it's 10 tokens long
    embeddings = np.random.randn(10, 64)
    token_counts = [1] * 10
    total_tokens = 10
    medoid_idx = 5
    return ClusterSegment(
        embeddings=embeddings,
        token_counts=token_counts,
        total_tokens=total_tokens,
        medoid_idx=medoid_idx
    )

def test_medoid_k_rope_3d(mock_segment):
    """ Test Positional Harvesting for 3D tensors: (batch, seq, dim) """
    batch_size = 2
    seq_len = 10
    dim = 128
    k_rope_3d = torch.randn(batch_size, seq_len, dim)
    c_kv_3d = torch.randn(batch_size, seq_len, dim)  # dummy
    
    cluster = SealedCluster(c_kv=c_kv_3d, k_rope=k_rope_3d, segment=mock_segment)
    
    # 1. Test medoid_k_rope (Harvesting)
    medoid_rope = cluster.medoid_k_rope
    assert medoid_rope.shape == (batch_size, 1, dim)
    assert torch.equal(medoid_rope, k_rope_3d[:, mock_segment.medoid_idx:mock_segment.medoid_idx+1, :])
    
    # 2. Test expand_medoid_k_rope (Injection)
    K = 4
    expanded_rope = cluster.expand_medoid_k_rope(K)
    assert expanded_rope.shape == (batch_size, K, dim)
    
    # Check that all elements along the seq dim are exactly the medoid slice
    for i in range(K):
        assert torch.equal(expanded_rope[:, i:i+1, :], medoid_rope)

def test_medoid_k_rope_4d(mock_segment):
    """ Test Positional Harvesting for 4D tensors: (batch, heads, seq, dim) """
    batch_size = 1
    num_heads = 8
    seq_len = 10
    dim = 64
    k_rope_4d = torch.randn(batch_size, num_heads, seq_len, dim)
    c_kv_4d = torch.randn(batch_size, num_heads, seq_len, dim)  # dummy
    
    cluster = SealedCluster(c_kv=c_kv_4d, k_rope=k_rope_4d, segment=mock_segment)
    
    # 1. Test medoid_k_rope (Harvesting)
    medoid_rope = cluster.medoid_k_rope
    assert medoid_rope.shape == (batch_size, num_heads, 1, dim)
    assert torch.equal(medoid_rope, k_rope_4d[:, :, mock_segment.medoid_idx:mock_segment.medoid_idx+1, :])
    
    # 2. Test expand_medoid_k_rope (Injection)
    K = 2
    expanded_rope = cluster.expand_medoid_k_rope(K)
    assert expanded_rope.shape == (batch_size, num_heads, K, dim)
    
    # Check that all elements along the seq dim are exactly the medoid slice
    for i in range(K):
        assert torch.equal(expanded_rope[:, :, i:i+1, :], medoid_rope)
