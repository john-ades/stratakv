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
        new_len = c_kv.shape[seq_dim_c]
        
        if getattr(self, "k_buffer", None) is None:
            # Pre-allocate buffer
            k_shape = list(c_kv.shape)
            v_shape = list(k_rope.shape)
            k_shape[seq_dim_c] = max(self.capacity, new_len)
            v_shape[seq_dim_k] = max(self.capacity, new_len)
            
            self.k_buffer = torch.empty(k_shape, dtype=c_kv.dtype, device=c_kv.device)
            self.v_buffer = torch.empty(v_shape, dtype=k_rope.dtype, device=k_rope.device)
            self.k_cache = None
            self.v_cache = None

        total_len = self.seq_len + new_len
        evicted_c_kv = None
        evicted_k_rope = None

        if self.capacity > 0 and total_len > self.capacity:
            evict_len = total_len - self.capacity
            
            # Extract evicted items
            if evict_len <= self.seq_len:
                if seq_dim_c == 1:
                    evicted_c_kv = self.k_buffer[:, :evict_len, ...].clone()
                else:
                    evicted_c_kv = self.k_buffer[:, :, :evict_len, ...].clone()
                
                if seq_dim_k == 1:
                    evicted_k_rope = self.v_buffer[:, :evict_len, ...].clone()
                else:
                    evicted_k_rope = self.v_buffer[:, :, :evict_len, ...].clone()
            else:
                # new_len exceeds capacity
                if seq_dim_c == 1:
                    evicted_c_new = c_kv[:, :evict_len - self.seq_len, ...]
                    evicted_c_kv = torch.cat([self.k_buffer[:, :self.seq_len, ...], evicted_c_new], dim=seq_dim_c) if self.seq_len > 0 else evicted_c_new
                else:
                    evicted_c_new = c_kv[:, :, :evict_len - self.seq_len, ...]
                    evicted_c_kv = torch.cat([self.k_buffer[:, :, :self.seq_len, ...], evicted_c_new], dim=seq_dim_c) if self.seq_len > 0 else evicted_c_new
                    
                if seq_dim_k == 1:
                    evicted_k_new = k_rope[:, :evict_len - self.seq_len, ...]
                    evicted_k_rope = torch.cat([self.v_buffer[:, :self.seq_len, ...], evicted_k_new], dim=seq_dim_k) if self.seq_len > 0 else evicted_k_new
                else:
                    evicted_k_new = k_rope[:, :, :evict_len - self.seq_len, ...]
                    evicted_k_rope = torch.cat([self.v_buffer[:, :, :self.seq_len, ...], evicted_k_new], dim=seq_dim_k) if self.seq_len > 0 else evicted_k_new

            # Shift remaining buffer left
            kept_len = max(0, self.seq_len - evict_len)
            if kept_len > 0:
                if seq_dim_c == 1:
                    self.k_buffer[:, :kept_len, ...] = self.k_buffer[:, evict_len:self.seq_len, ...].clone()
                else:
                    self.k_buffer[:, :, :kept_len, ...] = self.k_buffer[:, :, evict_len:self.seq_len, ...].clone()
                    
                if seq_dim_k == 1:
                    self.v_buffer[:, :kept_len, ...] = self.v_buffer[:, evict_len:self.seq_len, ...].clone()
                else:
                    self.v_buffer[:, :, :kept_len, ...] = self.v_buffer[:, :, evict_len:self.seq_len, ...].clone()
            
            self.seq_len = kept_len

        # Append new items
        take_new_len = min(new_len, self.capacity) if self.capacity > 0 else new_len
        start_new = max(0, new_len - self.capacity) if self.capacity > 0 else 0
        
        if seq_dim_c == 1:
            self.k_buffer[:, self.seq_len:self.seq_len+take_new_len, ...] = c_kv[:, start_new:, ...]
        else:
            self.k_buffer[:, :, self.seq_len:self.seq_len+take_new_len, ...] = c_kv[:, :, start_new:, ...]
            
        if seq_dim_k == 1:
            self.v_buffer[:, self.seq_len:self.seq_len+take_new_len, ...] = k_rope[:, start_new:, ...]
        else:
            self.v_buffer[:, :, self.seq_len:self.seq_len+take_new_len, ...] = k_rope[:, :, start_new:, ...]
        
        self.seq_len += take_new_len
        
        # update the cache views
        if seq_dim_c == 1:
            self.k_cache = self.k_buffer[:, :self.seq_len, ...]
        else:
            self.k_cache = self.k_buffer[:, :, :self.seq_len, ...]
            
        if seq_dim_k == 1:
            self.v_cache = self.v_buffer[:, :self.seq_len, ...]
        else:
            self.v_cache = self.v_buffer[:, :, :self.seq_len, ...]
        
        return evicted_c_kv, evicted_k_rope
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns (C_kv, K_rope)
        """
        return self.k_cache, self.v_cache
