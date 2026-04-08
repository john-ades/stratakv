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
        new_len = keys.shape[2]
        
        if getattr(self, "k_buffer", None) is None:
            # Pre-allocate buffer
            k_shape = list(keys.shape)
            v_shape = list(values.shape)
            k_shape[2] = max(self.capacity, new_len)
            v_shape[2] = max(self.capacity, new_len)
            
            self.k_buffer = torch.empty(k_shape, dtype=keys.dtype, device=keys.device)
            self.v_buffer = torch.empty(v_shape, dtype=values.dtype, device=values.device)
            self.k_cache = None
            self.v_cache = None

        total_len = self.seq_len + new_len
        evicted_keys = None
        evicted_values = None

        if total_len > self.capacity:
            evict_len = total_len - self.capacity
            
            # Extract evicted items
            if evict_len <= self.seq_len:
                evicted_keys = self.k_buffer[:, :, :evict_len, :].clone()
                evicted_values = self.v_buffer[:, :, :evict_len, :].clone()
            else:
                # new_len itself exceeds capacity! Evict everything in buffer + start of incoming
                evicted_k_new = keys[:, :, :evict_len - self.seq_len, :]
                evicted_v_new = values[:, :, :evict_len - self.seq_len, :]
                if self.seq_len > 0:
                    evicted_keys = torch.cat([self.k_buffer[:, :, :self.seq_len, :], evicted_k_new], dim=2)
                    evicted_values = torch.cat([self.v_buffer[:, :, :self.seq_len, :], evicted_v_new], dim=2)
                else:
                    evicted_keys = evicted_k_new
                    evicted_values = evicted_v_new

            # Shift remaining buffer left
            kept_len = max(0, self.seq_len - evict_len)
            if kept_len > 0:
                # use clone to avoid in-place overlapping memory anomalies 
                self.k_buffer[:, :, :kept_len, :] = self.k_buffer[:, :, evict_len:self.seq_len, :].clone()
                self.v_buffer[:, :, :kept_len, :] = self.v_buffer[:, :, evict_len:self.seq_len, :].clone()
            
            self.seq_len = kept_len

        # Now append the new (or remaining new) items to the open space
        take_new_len = min(new_len, self.capacity)
        start_new = max(0, new_len - self.capacity)
        
        self.k_buffer[:, :, self.seq_len:self.seq_len+take_new_len, :] = keys[:, :, start_new:, :]
        self.v_buffer[:, :, self.seq_len:self.seq_len+take_new_len, :] = values[:, :, start_new:, :]
        
        self.seq_len += take_new_len
        
        # update the cache views
        self.k_cache = self.k_buffer[:, :, :self.seq_len, :]
        self.v_cache = self.v_buffer[:, :, :self.seq_len, :]
        
        return evicted_keys, evicted_values
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.k_cache, self.v_cache
