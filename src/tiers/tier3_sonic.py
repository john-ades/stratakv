import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier3Sonic(StrataTier):
    """
    Tier 3 Orthogonal Sequence Cache (SONIC).
    This tier stores highly compressed Nexus tokens: C_nexus and the Medoid K_rope anchor.
    Expected shapes are [batch, k, dim] or [batch, num_heads, k, dim].
    """
    def push(self, c_nexus: torch.Tensor, k_rope_medoid: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Pushes Nexus tokens and their Medoid positional anchors into Tier 3.
        
        Args:
            c_nexus: The sequence-compressed attention representation of the cluster.
            k_rope_medoid: The harvested Medoid positional anchor token (K injected).
        
        Returns:
            Tuple of (evicted_c_nexus, evicted_k_rope) if capacity exceeded. 
            Tier 3 usually has very large capacity relative to T1/T2, but eviction must still be handled.
        """
        seq_dim_c = 1 if c_nexus.dim() == 3 else 2
        seq_dim_k = 1 if k_rope_medoid.dim() == 3 else 2
        
        # In Tier 3, C_nexus and K_rope should have identical sequence lengths representing K
        assert c_nexus.shape[seq_dim_c] == k_rope_medoid.shape[seq_dim_k], \
            f"Sequence dimensions must match. got {c_nexus.shape[seq_dim_c]} and {k_rope_medoid.shape[seq_dim_k]}"
            
        # Append incoming to existing
        if self.k_cache is None:
            self.k_cache = c_nexus
            self.v_cache = k_rope_medoid
        else:
            self.k_cache = torch.cat([self.k_cache, c_nexus], dim=seq_dim_c)
            self.v_cache = torch.cat([self.v_cache, k_rope_medoid], dim=seq_dim_k)
            
        self.seq_len = self.k_cache.shape[seq_dim_c]
        
        # If capacity > 0 and we exceed it, evict earliest part of the cache
        if self.capacity > 0 and self.seq_len > self.capacity:
            evict_len = self.seq_len - self.capacity
            
            if seq_dim_c == 1:
                evicted_c_nexus = self.k_cache[:, :evict_len, ...]
                self.k_cache = self.k_cache[:, evict_len:, ...]
            else:
                evicted_c_nexus = self.k_cache[:, :, :evict_len, ...]
                self.k_cache = self.k_cache[:, :, evict_len:, ...]
                
            if seq_dim_k == 1:
                evicted_k_rope = self.v_cache[:, :evict_len, ...]
                self.v_cache = self.v_cache[:, evict_len:, ...]
            else:
                evicted_k_rope = self.v_cache[:, :, :evict_len, ...]
                self.v_cache = self.v_cache[:, :, evict_len:, ...]
                
            self.seq_len = self.capacity
            return evicted_c_nexus, evicted_k_rope
            
        return None, None
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (C_nexus, K_rope_medoids)
        """
        return self.k_cache, self.v_cache
