import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier2Latent(StrataTier):
    def push(self, c_kv: torch.Tensor, k_rope: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        seq_dim_c = 1 if c_kv.dim() == 3 else 2
        seq_dim_k = 1 if k_rope.dim() == 3 else 2
        new_len = c_kv.shape[seq_dim_c]
        
        if getattr(self, "k_cache", None) is None:
            self.k_cache, self.v_cache, self.seq_len = c_kv, k_rope, new_len
        else:
            self.k_cache = torch.cat([self.k_cache, c_kv], dim=seq_dim_c)
            self.v_cache = torch.cat([self.v_cache, k_rope], dim=seq_dim_k)
            self.seq_len += new_len

        evicted_c_kv, evicted_k_rope = None, None

        if self.capacity > 0 and self.seq_len > self.capacity:
            evict_len = self.seq_len - self.capacity
            
            if seq_dim_c == 1:
                evicted_c_kv = self.k_cache[:, :evict_len, ...]
                self.k_cache = self.k_cache[:, evict_len:, ...]
            else:
                evicted_c_kv = self.k_cache[:, :, :evict_len, ...]
                self.k_cache = self.k_cache[:, :, evict_len:, ...]
                
            if seq_dim_k == 1:
                evicted_k_rope = self.v_cache[:, :evict_len, ...]
                self.v_cache = self.v_cache[:, evict_len:, ...]
            else:
                evicted_k_rope = self.v_cache[:, :, :evict_len, ...]
                self.v_cache = self.v_cache[:, :, evict_len:, ...]
                
            self.seq_len = self.capacity

        return evicted_c_kv, evicted_k_rope
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return getattr(self, "k_cache", None), getattr(self, "v_cache", None)
