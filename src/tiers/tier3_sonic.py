import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier3Sonic(StrataTier):
    def __init__(self, capacity: int, layer_idx: int):
        super().__init__(capacity, layer_idx)
        self.k_caches = {}
        self.v_caches = {}
        self.max_seq_len = 0

    def push(self, c_nexus: torch.Tensor, k_rope_medoid: torch.Tensor, batch_idx: int = 0, batch_size: int = 1) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        seq_dim_c = 1 if c_nexus.dim() == 3 else 2
        seq_dim_k = 1 if k_rope_medoid.dim() == 3 else 2
        
        c_nexus_b = c_nexus[batch_idx:batch_idx+1, ...]
        k_rope_b = k_rope_medoid[batch_idx:batch_idx+1, ...]
        
        if batch_idx not in self.k_caches:
            self.k_caches[batch_idx] = c_nexus_b
            self.v_caches[batch_idx] = k_rope_b
        else:
            self.k_caches[batch_idx] = torch.cat([self.k_caches[batch_idx], c_nexus_b], dim=seq_dim_c)
            self.v_caches[batch_idx] = torch.cat([self.v_caches[batch_idx], k_rope_b], dim=seq_dim_k)
            
        current_len = self.k_caches[batch_idx].shape[seq_dim_c]
        evicted_c_nexus, evicted_k_rope = None, None
        
        if self.capacity > 0 and current_len > self.capacity:
            evict_len = current_len - self.capacity
            
            if seq_dim_c == 1:
                evicted_c_nexus = self.k_caches[batch_idx][:, :evict_len, ...]
                self.k_caches[batch_idx] = self.k_caches[batch_idx][:, evict_len:, ...]
            else:
                evicted_c_nexus = self.k_caches[batch_idx][:, :, :evict_len, ...]
                self.k_caches[batch_idx] = self.k_caches[batch_idx][:, :, evict_len:, ...]
                
            if seq_dim_k == 1:
                evicted_k_rope = self.v_caches[batch_idx][:, :evict_len, ...]
                self.v_caches[batch_idx] = self.v_caches[batch_idx][:, evict_len:, ...]
            else:
                evicted_k_rope = self.v_caches[batch_idx][:, :, :evict_len, ...]
                self.v_caches[batch_idx] = self.v_caches[batch_idx][:, :, evict_len:, ...]

        self.max_seq_len = max([t.shape[1 if t.dim() == 3 else 2] for t in self.k_caches.values()])
        return evicted_c_nexus, evicted_k_rope
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not self.k_caches or self.max_seq_len == 0:
            return None, None
            
        batch_size = max(self.k_caches.keys()) + 1
        padded_k, padded_v = [], []
        
        first_c = next(iter(self.k_caches.values()))
        first_k = next(iter(self.v_caches.values()))
        seq_dim_c = 1 if first_c.dim() == 3 else 2
        seq_dim_k = 1 if first_k.dim() == 3 else 2
        
        for b in range(batch_size):
            if b in self.k_caches:
                c_t, k_t = self.k_caches[b], self.v_caches[b]
                pad_len = self.max_seq_len - c_t.shape[seq_dim_c]
                if pad_len > 0:
                    pad_shape_c, pad_shape_k = list(c_t.shape), list(k_t.shape)
                    pad_shape_c[seq_dim_c], pad_shape_k[seq_dim_k] = pad_len, pad_len
                    c_t = torch.cat([c_t, torch.zeros(pad_shape_c, dtype=c_t.dtype, device=c_t.device)], dim=seq_dim_c)
                    k_t = torch.cat([k_t, torch.zeros(pad_shape_k, dtype=k_t.dtype, device=k_t.device)], dim=seq_dim_k)
                padded_k.append(c_t)
                padded_v.append(k_t)
            else:
                shape_c, shape_k = list(first_c.shape), list(first_k.shape)
                shape_c[0], shape_k[0] = 1, 1 
                shape_c[seq_dim_c], shape_k[seq_dim_k] = self.max_seq_len, self.max_seq_len
                padded_k.append(torch.zeros(shape_c, dtype=first_c.dtype, device=first_c.device))
                padded_v.append(torch.zeros(shape_k, dtype=first_k.dtype, device=first_k.device))
                
        return torch.cat(padded_k, dim=0), torch.cat(padded_v, dim=0)
