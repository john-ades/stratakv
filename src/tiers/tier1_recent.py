import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier1Recent(StrataTier):
    """
    Tier 1 Recent Cache (Sliding Window).
    This tier holds the `capacity` most recent tokens.
    When new tokens push the total beyond capacity, the *oldest* tokens are evicted.
    """
    
    def push(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # keys/values shape expected: [batch_size, num_heads, seq_len, head_dim]
        
        # Append incoming to existing
        if self.k_cache is None:
            self.k_cache = keys
            self.v_cache = values
        else:
            self.k_cache = torch.cat([self.k_cache, keys], dim=2)
            self.v_cache = torch.cat([self.v_cache, values], dim=2)
            
        self.seq_len = self.k_cache.shape[2]
        
        # If we exceed capacity, evict the earliest part of the cache
        if self.seq_len > self.capacity:
            evict_len = self.seq_len - self.capacity
            
            evicted_keys = self.k_cache[:, :, :evict_len, :]
            evicted_values = self.v_cache[:, :, :evict_len, :]
            
            self.k_cache = self.k_cache[:, :, evict_len:, :]
            self.v_cache = self.v_cache[:, :, evict_len:, :]
            self.seq_len = self.capacity
            
            return evicted_keys, evicted_values
            
        return None, None
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.k_cache, self.v_cache
