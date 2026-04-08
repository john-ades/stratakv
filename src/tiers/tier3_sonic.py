import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier3Sonic(StrataTier):
    """
    Tier 3 Orthogonal Sequence Cache (SONIC).
    This tier stores highly compressed Nexus tokens: C_nexus and the Medoid K_rope anchor.
    Expected shapes are [batch, k, dim] or [batch, num_heads, k, dim].
    """
    def push(self, c_nexus: torch.Tensor, k_rope_medoid: torch.Tensor, batch_idx: int = 0, batch_size: int = 1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        seq_dim_c = 1 if c_nexus.dim() == 3 else 2
        seq_dim_k = 1 if k_rope_medoid.dim() == 3 else 2
        
        # In Tier 3, C_nexus and K_rope should have identical sequence lengths representing K
        assert c_nexus.shape[seq_dim_c] == k_rope_medoid.shape[seq_dim_k]
            
        new_len = c_nexus.shape[seq_dim_c]
        
        if getattr(self, "k_buffer", None) is None:
            # Pre-allocate buffer
            k_shape = list(c_nexus.shape)
            v_shape = list(k_rope_medoid.shape)
            
            # Reconstruct batch dimension
            k_shape[0] = batch_size
            v_shape[0] = batch_size
            
            k_shape[seq_dim_c] = max(self.capacity, new_len) if self.capacity > 0 else 1024
            v_shape[seq_dim_k] = max(self.capacity, new_len) if self.capacity > 0 else 1024
            
            self.k_buffer = torch.zeros(k_shape, dtype=c_nexus.dtype, device=c_nexus.device)
            self.v_buffer = torch.zeros(v_shape, dtype=k_rope_medoid.dtype, device=k_rope_medoid.device)
            self.seq_lens = [0] * batch_size
            self.max_seq_len = 0
            
        # Write to specific batch_idx
        idx_seq_len = self.seq_lens[batch_idx]
        
        # Simple capacity check
        evicted_c_nexus, evicted_k_rope = None, None
        
        if self.capacity > 0 and idx_seq_len + new_len > self.capacity:
            evict_len = (idx_seq_len + new_len) - self.capacity
            
            # Extract evicted items
            if evict_len <= idx_seq_len:
                if seq_dim_c == 1:
                    evicted_c_nexus = self.k_buffer[batch_idx:batch_idx+1, :evict_len, ...].clone()
                else:
                    evicted_c_nexus = self.k_buffer[batch_idx:batch_idx+1, :, :evict_len, ...].clone()
                
                if seq_dim_k == 1:
                    evicted_k_rope = self.v_buffer[batch_idx:batch_idx+1, :evict_len, ...].clone()
                else:
                    evicted_k_rope = self.v_buffer[batch_idx:batch_idx+1, :, :evict_len, ...].clone()
            else:
                # new_len itself exceeds capacity
                if seq_dim_c == 1:
                    evicted_c_new = c_nexus[:, :evict_len - idx_seq_len, ...]
                    evicted_c_nexus = torch.cat([self.k_buffer[batch_idx:batch_idx+1, :idx_seq_len, ...], evicted_c_new], dim=seq_dim_c) if idx_seq_len > 0 else evicted_c_new
                else:
                    evicted_c_new = c_nexus[:, :, :evict_len - idx_seq_len, ...]
                    evicted_c_nexus = torch.cat([self.k_buffer[batch_idx:batch_idx+1, :, :idx_seq_len, ...], evicted_c_new], dim=seq_dim_c) if idx_seq_len > 0 else evicted_c_new
                    
                if seq_dim_k == 1:
                    evicted_k_new = k_rope_medoid[:, :evict_len - idx_seq_len, ...]
                    evicted_k_rope = torch.cat([self.v_buffer[batch_idx:batch_idx+1, :idx_seq_len, ...], evicted_k_new], dim=seq_dim_k) if idx_seq_len > 0 else evicted_k_new
                else:
                    evicted_k_new = k_rope_medoid[:, :, :evict_len - idx_seq_len, ...]
                    evicted_k_rope = torch.cat([self.v_buffer[batch_idx:batch_idx+1, :, :idx_seq_len, ...], evicted_k_new], dim=seq_dim_k) if idx_seq_len > 0 else evicted_k_new
            
            # Shift remaining buffer left
            kept_len = max(0, idx_seq_len - evict_len)
            if kept_len > 0:
                if seq_dim_c == 1:
                    self.k_buffer[batch_idx:batch_idx+1, :kept_len, ...] = self.k_buffer[batch_idx:batch_idx+1, evict_len:idx_seq_len, ...].clone()
                else:
                    self.k_buffer[batch_idx:batch_idx+1, :, :kept_len, ...] = self.k_buffer[batch_idx:batch_idx+1, :, evict_len:idx_seq_len, ...].clone()
                    
                if seq_dim_k == 1:
                    self.v_buffer[batch_idx:batch_idx+1, :kept_len, ...] = self.v_buffer[batch_idx:batch_idx+1, evict_len:idx_seq_len, ...].clone()
                else:
                    self.v_buffer[batch_idx:batch_idx+1, :, :kept_len, ...] = self.v_buffer[batch_idx:batch_idx+1, :, evict_len:idx_seq_len, ...].clone()
            
            self.seq_lens[batch_idx] = kept_len
            idx_seq_len = kept_len
            take_len = min(new_len, self.capacity)
            start_new = max(0, new_len - self.capacity)
        else:
            take_len = new_len
            start_new = 0
            
        if take_len > 0:
            if seq_dim_c == 1:
                self.k_buffer[batch_idx:batch_idx+1, idx_seq_len:idx_seq_len+take_len, ...] = c_nexus[:, start_new:, ...]
            else:
                self.k_buffer[batch_idx:batch_idx+1, :, idx_seq_len:idx_seq_len+take_len, ...] = c_nexus[:, :, start_new:, ...]
                
            if seq_dim_k == 1:
                self.v_buffer[batch_idx:batch_idx+1, idx_seq_len:idx_seq_len+take_len, ...] = k_rope_medoid[:, start_new:, ...]
            else:
                self.v_buffer[batch_idx:batch_idx+1, :, idx_seq_len:idx_seq_len+take_len, ...] = k_rope_medoid[:, :, start_new:, ...]
                
            self.seq_lens[batch_idx] += take_len
            self.max_seq_len = max(self.seq_lens)
            
        return evicted_c_nexus, evicted_k_rope
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if getattr(self, "k_buffer", None) is None or self.max_seq_len == 0:
            return None, None
            
        seq_dim_c = 1 if self.k_buffer.dim() == 3 else 2
        seq_dim_k = 1 if self.v_buffer.dim() == 3 else 2
        
        if seq_dim_c == 1:
            k_cache = self.k_buffer[:, :self.max_seq_len, ...]
        else:
            k_cache = self.k_buffer[:, :, :self.max_seq_len, ...]
            
        if seq_dim_k == 1:
            v_cache = self.v_buffer[:, :self.max_seq_len, ...]
        else:
            v_cache = self.v_buffer[:, :, :self.max_seq_len, ...]
            
        return k_cache, v_cache
