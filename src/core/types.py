import torch
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class TokenState:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    seq_len: int
