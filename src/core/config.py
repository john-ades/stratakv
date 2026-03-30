from dataclasses import dataclass

@dataclass
class StrataKVConfig:
    tier0_size: int = 4
    tier1_size: int = 2048
    tier2_size: int = 8192
    enable_tier0: bool = True
    enable_tier1: bool = True
    enable_tier2: bool = False
    enable_tier3: bool = False
    
    # Tier 2 TransMLA Parameters
    num_kv_heads: int = 8
    head_dim: int = 128
    transmla_target_rank: int = 32
    transmla_rope_dim: int = 64
    transmla_matrices_path: str = None
