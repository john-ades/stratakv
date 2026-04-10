import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier1Recent(StrataTier):
    def push(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        new_len = keys.shape[2]
        
        if getattr(self, "k_cache", None) is None:
            self.k_cache, self.v_cache, self.seq_len = keys, values, new_len
        else:
            self.k_cache = torch.cat([self.k_cache, keys], dim=2)
            self.v_cache = torch.cat([self.v_cache, values], dim=2)
            self.seq_len += new_len

        evicted_keys, evicted_values = None, None

        if self.capacity > 0 and self.seq_len > self.capacity:
            evict_len = self.seq_len - self.capacity
            
            evicted_keys = self.k_cache[:, :, :evict_len, :]
            evicted_values = self.v_cache[:, :, :evict_len, :]
            
            self.k_cache = self.k_cache[:, :, evict_len:, :]
            self.v_cache = self.v_cache[:, :, evict_len:, :]
            self.seq_len = self.capacity

        return evicted_keys, evicted_values
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return getattr(self, "k_cache", None), getattr(self, "v_cache", None)
