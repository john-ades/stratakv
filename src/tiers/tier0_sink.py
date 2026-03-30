import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier0Sink(StrataTier):
    """
    Tier 0 Sink Cache.
    This tier captures the first `capacity` tokens it receives and holds them permanently.
    Any tokens beyond its capacity are immediately yielded out to be passed to deeper tiers.
    """
    
    def push(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # keys/values shape expected: [batch_size, num_heads, incoming_seq_len, head_dim]
        # We assume batch size and other dims are consistent
        
        incoming_seq_len = keys.shape[2]
        space_left = self.capacity - self.seq_len
        
        if space_left <= 0:
            # We are full, all incoming tokens are evicted out to the next tier immediately
            return keys, values
            
        # We have some space left
        take_len = min(space_left, incoming_seq_len)
        
        # Take the slice to store
        keys_to_store = keys[:, :, :take_len, :]
        values_to_store = values[:, :, :take_len, :]
        
        if self.k_cache is None:
            self.k_cache = keys_to_store
            self.v_cache = values_to_store
        else:
            self.k_cache = torch.cat([self.k_cache, keys_to_store], dim=2)
            self.v_cache = torch.cat([self.v_cache, values_to_store], dim=2)
            
        self.seq_len += take_len
        
        # Remaining tokens to push down
        if take_len < incoming_seq_len:
            evicted_keys = keys[:, :, take_len:, :]
            evicted_values = values[:, :, take_len:, :]
            return evicted_keys, evicted_values
            
        return None, None
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.k_cache, self.v_cache
