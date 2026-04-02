import pytest
import torch
import concurrent.futures
from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache

def simulate_user_request(cache: StrataKVCache, user_id: int):
    """
    Simulates a single user concurrently writing to the cache across 2 layers.
    Due to Python's GIL, thread interleaving can cause race conditions if the cache 
    isn't robust or relies on unstable state loops.
    """
    batch_size = 1
    num_heads = 4
    seq_len = 8
    head_dim = 16
    
    # We make the tensor values uniquely identifiable for this user
    k_states = torch.ones(batch_size, num_heads, seq_len, head_dim) * user_id
    v_states = torch.ones(batch_size, num_heads, seq_len, head_dim) * user_id
    
    # Push to layer 0
    k_out_0, v_out_0 = cache.update(k_states, v_states, layer_idx=0)
    
    # Push to layer 1
    k_out_1, v_out_1 = cache.update(k_states, v_states, layer_idx=1)
    
    return user_id, k_out_0, v_out_0, k_out_1, v_out_1

def test_cache_multi_tenancy_concurrency():
    """
    Hits the StrataKVCache with multiple concurrent threads to ensure
    memory updates don't cause dimension mismatch or tensor corruption under load.
    Note: Thread-safety is NOT guaranteed by default in PyTorch tensor ops without locks,
    so this test highlights bounds/synchronization needs for multitenancy.
    """
    config = StrataKVConfig(
        tier0_size=4,
        tier1_size=8,
        tier2_size=16,
        enable_tier0=True,
        enable_tier1=True, # Will cause eviction
        enable_tier2=False # Keep it simple for strict threading test
    )
    
    cache = StrataKVCache(config)
    cache._ensure_initialized(1) # layer 0 and 1
    
    num_users = 10
    
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(simulate_user_request, cache, i) for i in range(num_users)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                user_id, k0, v0, k1, v1 = future.result()
                success_count += 1
            except Exception as e:
                # If race conditions occur, cache update might fail 
                # (e.g., list append during iteration, dim mismatch during torch.cat)
                pytest.fail(f"Concurrency error on user {user_id}: {str(e)}")
                
    assert success_count == num_users, "Not all simulated user requests succeeded."
    
    # Verify the cache size maxed out properly and didn't overflow bounds due to races
    layer_0_len = cache.get_seq_length(0)
    # the max size it can hold is tier0_size + tier1_size = 12
    # Since 10 users * 8 seq_len = 80 tokens, it should heavily truncate and evict.
    assert layer_0_len <= 12, f"Cache exceeded memory bounds! Length: {layer_0_len}"
