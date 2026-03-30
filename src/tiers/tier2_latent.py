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
        # Determine sequence length dimension based on tensor rank:
        # If [batch, seq_len, dim], dim=1
        # If [batch, heads, seq_len, dim], dim=2
        seq_dim = 1 if c_kv.dim() == 3 else 2
        
        # Append incoming to existing
        if self.k_cache is None:
             # c_kv replaces keys
            self.k_cache = c_kv
            # k_rope replaces values
            self.v_cache = k_rope
        else:
            self.k_cache = torch.cat([self.k_cache, c_kv], dim=seq_dim)
            self.v_cache = torch.cat([self.v_cache, k_rope], dim=seq_dim)
            
        self.seq_len = self.k_cache.shape[seq_dim]
        
        # If capacity > 0 and we exceed it, evict earliest part of the cache
        if self.capacity > 0 and self.seq_len > self.capacity:
            evict_len = self.seq_len - self.capacity
            
            if seq_dim == 1:
                evicted_c_kv = self.k_cache[:, :evict_len, ...]
                evicted_k_rope = self.v_cache[:, :evict_len, ...]
                
                self.k_cache = self.k_cache[:, evict_len:, ...]
                self.v_cache = self.v_cache[:, evict_len:, ...]
            else:
                evicted_c_kv = self.k_cache[:, :, :evict_len, ...]
                evicted_k_rope = self.v_cache[:, :, :evict_len, ...]
                
                self.k_cache = self.k_cache[:, :, evict_len:, ...]
                self.v_cache = self.v_cache[:, :, evict_len:, ...]
                
            self.seq_len = self.capacity
            return evicted_c_kv, evicted_k_rope
            
        return None, None
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (C_kv, K_rope)
        """
        return self.k_cache, self.v_cache
