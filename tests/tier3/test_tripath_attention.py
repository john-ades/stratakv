import torch
import torch.nn as nn
from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
from src.compression.transmla import TransMLAAbsorber, TransMLACruncher
from src.compression.sonic import SonicCruncher
from src.models.llama.modeling_llama import patch_llama_for_strata

class LlamaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer_idx = 0
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.attention_dropout = 0.0

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=False)
        
    def forward(self, *args, **kwargs):
        pass # Will be monkey patched
        
    @property
    def dtype(self):
        return self.q_proj.weight.dtype
        
    @property
    def device(self):
        return self.q_proj.weight.device

def test_tripath_attention():
    # 1. Setup minimal configuration
    config = StrataKVConfig()
    config.hidden_size = 64
    config.num_attention_heads = 4
    config.num_kv_heads = 2
    config.num_key_value_heads = 2
    config.head_dim = 16
    config.enable_tier0 = False
    config.enable_tier1 = True
    config.tier1_size = 4
    config.enable_tier2 = True
    config.tier2_size = 8
    config.transmla_target_rank = 12
    config.transmla_rope_dim = 8
    config.enable_tier3 = True
    config.tier3_size = 128
    config.tier3_max_k = 2
    config.abit_window_size = 2
    config.abit_min_split_tokens = 2
    config.abit_max_split_tokens = 4
    config.tier3_k = 1

    # 2. Patch a Mock Llama Attention Module
    attn = LlamaAttention(config)
    patch_llama_for_strata(attn, config)

    # 3. Setup inputs
    batch_size = 1
    seq_len = 5 # Should trigger tier1 spillover into tier2, and eventually buffer boundary
    
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    
    # Dummy RoPE embeddings
    cos = torch.randn(batch_size, seq_len, config.head_dim)
    sin = torch.randn(batch_size, seq_len, config.head_dim)
    
    cache = StrataKVCache(config)
    
    # Simulate a few autoregressive generation steps to build up the cache pipeline
    # Step 1: Pre-fill some tokens
    out, _ = attn(
        hidden_states=hidden_states[:, :3, :],
        position_embeddings=(cos[:, :3, :], sin[:, :3, :]),
        past_key_values=cache,
        use_cache=True
    )
    
    # Assert tier 1 caught them
    t1 = cache._tier1_recents[0].k_cache
    assert t1.size(2) == 3 
    assert cache._tier2_latents[0].k_cache is None
    
    # Step 2: More tokens to trigger spillover to T2 and then seal cluster to T3
    for i in range(3, 10):
        # We process 1 token at a time
        hidden_t = torch.randn(1, 1, config.hidden_size)
        cos_t = cos[:, :1, :]
        sin_t = sin[:, :1, :]
        
        out, _ = attn(
            hidden_states=hidden_t,
            position_embeddings=(cos_t, sin_t),
            past_key_values=cache,
            use_cache=True
        )

    # Verify T1 has exactly `tier1_size`
    assert cache._tier1_recents[0].k_cache.size(2) == config.tier1_size
    
    # Verify T2 has tokens
    assert cache._tier2_latents[0].k_cache is not None

    # Verify T3 got something because min_split is 2 and we pushed 6 tokens to buffer
    c_kv_t3, k_rope_t3 = cache.get_tier3_cache(0)
    assert c_kv_t3 is not None, "Tier 3 cache should not be strictly empty"
    assert k_rope_t3 is not None
    
    # Run a final forward pass over the full tri-path cache
    hidden_query = torch.randn(1, 1, config.hidden_size)
    
    out, weights = attn(
        hidden_states=hidden_query,
        position_embeddings=(cos[:, :1, :], sin[:, :1, :]),
        past_key_values=cache,
        use_cache=True,
        output_attentions=True
    )
    
    # The output should project to hidden size cleanly
    assert out.shape == (1, 1, config.hidden_size)
    
    # The Attention weights should cover T1 + T2 + T3 lengths!
    T1_len = cache._tier1_recents[0].seq_len
    T2_len = cache._tier2_latents[0].seq_len
    T3_len = cache._tier3_sonics[0].max_seq_len
    
    assert weights.size(-1) == T1_len + T2_len + T3_len, (
        f"Weights size {weights.size(-1)} != expected total {T1_len + T2_len + T3_len}"
    )

if __name__ == "__main__":
    test_tripath_attention()
