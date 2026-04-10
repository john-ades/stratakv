import torch
from typing import Optional, Tuple
from .base_tier import StrataTier

class Tier0Sink(StrataTier):
    def push(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        incoming_seq_len = keys.shape[2]
        space_left = self.capacity - self.seq_len
        
        if space_left <= 0: return keys, values
            
        take_len = min(space_left, incoming_seq_len)
        keys_to_store = keys[:, :, :take_len, :]
        values_to_store = values[:, :, :take_len, :]
        
        if getattr(self, "k_cache", None) is None:
            self.k_cache, self.v_cache = keys_to_store, values_to_store
        else:
            self.k_cache = torch.cat([self.k_cache, keys_to_store], dim=2)
            self.v_cache = torch.cat([self.v_cache, values_to_store], dim=2)
            
        self.seq_len += take_len
        
        if take_len < incoming_seq_len:
            return keys[:, :, take_len:, :], values[:, :, take_len:, :]
        return None, None
        
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return getattr(self, "k_cache", None), getattr(self, "v_cache", None)
