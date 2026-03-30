import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from transformers.cache_utils import Cache
try:
    from transformers.models.llama.modeling_llama import repeat_kv, apply_rotary_pos_emb
except ImportError:
    pass

from ...cache_manager import StrataKVCache
from ...core.config import StrataKVConfig
from ...compression.transmla import TransMLAAbsorber

def create_strata_llama_cache(config: StrataKVConfig) -> StrataKVCache:
    """
    Creates a StrataKVCache for Llama models.
    Pass this into `model.forward(..., past_key_values=cache)` 
    or `model.generate(..., past_key_values=cache)`.
    """
    return StrataKVCache(config)
    
def _strata_llama_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    num_heads = self.config.num_attention_heads
    num_key_value_heads = self.config.num_key_value_heads
    hidden_size = self.config.hidden_size

    if getattr(self.config, "pretraining_tp", 1) > 1:
        key_value_slicing = (num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [torch.nn.functional.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [torch.nn.functional.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [torch.nn.functional.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        pass # In Transformers 5.4+ position_embeddings should be passed.
    else:
        cos, sin = position_embeddings
    
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    
    if past_key_values is not None:
        cache_kwargs = {
            "sin": sin, 
            "cos": cos, 
            "cache_position": cache_position,
            "strata_cruncher": getattr(self, "strata_cruncher", None)
        }
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)
        
    key_states_t1 = repeat_kv(key_states, self.num_key_value_groups)
    value_states_t1 = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights_t1 = torch.matmul(query_states, key_states_t1.transpose(2, 3)) / math.sqrt(self.head_dim)
    
    attn_weights_t2 = None
    c_kv_t2, k_rope_t2 = None, None
    if isinstance(past_key_values, StrataKVCache) and hasattr(self, "strata_absorber"):
        c_kv_t2, k_rope_t2 = past_key_values.get_tier2_cache(self.layer_idx)
        
    if c_kv_t2 is not None and k_rope_t2 is not None:
        scores_t2_raw = self.strata_absorber.absorb_and_score(query_states, c_kv_t2, k_rope_t2)
        attn_weights_t2 = scores_t2_raw / math.sqrt(self.head_dim)
        
    if attn_weights_t2 is not None:
        attn_weights = torch.cat([attn_weights_t2, attn_weights_t1], dim=-1)
    else:
        attn_weights = attn_weights_t1

    if attention_mask is not None:
        if attention_mask.size(-1) == attn_weights.size(-1):
            attn_weights = attn_weights + attention_mask
        else:
            attn_weights = attn_weights + attention_mask[..., -attn_weights.size(-1):]

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights_dropped = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

    if attn_weights_t2 is not None:
        seq_len_t2 = c_kv_t2.size(1)
        w_t2, w_t1 = torch.split(attn_weights_dropped, [seq_len_t2, key_states.size(2)], dim=-1)
            
        out_t1 = torch.matmul(w_t1, value_states_t1)
        out_t2 = self.strata_absorber.decompress_value(w_t2, c_kv_t2)
        attn_output = out_t1 + out_t2
    else:
        attn_output = torch.matmul(attn_weights_dropped, value_states_t1)
        
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, hidden_size)
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights

def patch_llama_for_strata(model, config: StrataKVConfig):
    """
    Monkey-patch Llama model to use StrataKV hybrid attention.
    Instantiates a TransMLAAbsorber for each layer.
    """
    import types
    for name, module in model.named_modules():
        if module.__class__.__name__ == "LlamaAttention":
            if config.enable_tier2:
                absorber = TransMLAAbsorber(
                    layer_idx=module.layer_idx,
                    num_kv_heads=config.num_kv_heads,
                    head_dim=config.head_dim,
                    rope_retained_dim=config.transmla_rope_dim,
                    target_rank=config.transmla_target_rank,
                    matrices_path=config.transmla_matrices_path
                )
                module.add_module("strata_absorber", absorber)
                
                from ...compression.transmla import TransMLACruncher
                cruncher = TransMLACruncher(
                    layer_idx=module.layer_idx,
                    num_kv_heads=config.num_kv_heads,
                    head_dim=config.head_dim,
                    rope_retained_dim=config.transmla_rope_dim,
                    target_rank=config.transmla_target_rank,
                    matrices_path=config.transmla_matrices_path
                )
                module.add_module("strata_cruncher", cruncher)
                
            module.forward = types.MethodType(_strata_llama_attention_forward, module)
