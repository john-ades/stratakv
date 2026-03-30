from abc import ABC, abstractmethod
from typing import Optional, Tuple
import torch

class StrataTier(ABC):
    def __init__(self, capacity: int, layer_idx: int):
        self.capacity = capacity
        self.layer_idx = layer_idx
        # State: store keys and values
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self.seq_len: int = 0
    
    @abstractmethod
    def push(self, keys: torch.Tensor, values: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Pushes new tokens into the tier.
        Returns evicted (keys, values) if capacity is exceeded, else (None, None).
        Expected tensor shape: [batch_size, num_heads, seq_len, head_dim]
        """
        pass
        
    @abstractmethod
    def get_cache(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns the current stored (keys, values) for this tier.
        """
        pass
