from dataclasses import dataclass

@dataclass
class StrataKVConfig:
    tier0_size: int = 4
    tier1_size: int = 2048
    enable_tier0: bool = True
    enable_tier1: bool = True
    enable_tier2: bool = False
    enable_tier3: bool = False
