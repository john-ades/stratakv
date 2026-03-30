import torch
from ...cache_manager import StrataKVCache
from ...core.config import StrataKVConfig

def create_strata_llama_cache(config: StrataKVConfig) -> StrataKVCache:
    """
    Creates a StrataKVCache for Llama models.
    Pass this into `model.forward(..., past_key_values=cache)` 
    or `model.generate(..., past_key_values=cache)`.
    """
    return StrataKVCache(config)
    
def patch_llama_for_strata(model):
    """
    Optional: Monkey-patch Llama model if strict hijacking is needed
    instead of passing `past_key_values` manually. 
    Not fully required if using `transformers>=4.36`.
    """
    pass
