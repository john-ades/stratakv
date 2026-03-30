import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier2Latent(StrataTier):
    """
    Tier 2 Latent Cache (TransMLA Compressed).
    This tier stores highly compressed tokens: C_kv and K_rope.
    Expected shapes are [batch, seq_len, dim] or [batch, num_heads, seq_len, dim].
    """

    def push(self, c_kv: torch.Tensor, k_rope: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Pushes compressed representations into Tier 2.
        Because this is a latent format, we accept `c_kv` instead of `keys` and `k_rope` instead of `values`.
        """
        seq_dim_c = 1 if c_kv.dim() == 3 else 2
        seq_dim_k = 1 if k_rope.dim() == 3 else 2
        
        # Append incoming to existing
        if self.k_cache is None:
            self.k_cache = c_kv
            self.v_cache = k_rope
        else:
            self.k_cache = torch.cat([self.k_cache, c_kv], dim=seq_dim_c)
            self.v_cache = torch.cat([self.v_cache, k_rope], dim=seq_dim_k)
            
        self.seq_len = self.k_cache.shape[seq_dim_c]
        
        # If capacity > 0 and we exceed it, evict earliest part of the cache
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
            
        return None, None
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (C_kv, K_rope)
        """
        return self.k_cache, self.v_cache
