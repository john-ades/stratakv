from dataclasses import dataclass

DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B"

@dataclass
class StrataKVConfig:
    tier0_size: int = 4
    tier1_size: int = 2048
    tier2_size: int = 8192
    enable_tier0: bool = True
    enable_tier1: bool = True
    enable_tier2: bool = False
    enable_tier3: bool = False
    
    tier3_size: int = 65536
    tier3_k: int = 4
    tier3_max_k: int = 4
    
    # Tier 2 TransMLA Parameters
    num_kv_heads: int = 8
    head_dim: int = 128
    transmla_target_rank: int = 32
    transmla_rope_dim: int = 64
    transmla_matrices_path: str = None
    
    # Tier 3 ABIT Clustering Parameters
    abit_window_size: int = 3
    abit_min_split_tokens: int = 5
    abit_max_split_tokens: int = 128
    abit_initial_threshold: float = 0.5
