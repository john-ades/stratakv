import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from ...cache_manager import StrataKVCache
from ...core.config import StrataKVConfig
from ...compression.transmla import TransMLAAbsorber, TransMLACruncher
from .modeling_llama import patch_llama_for_strata

def prepare_for_healing(model: LlamaForCausalLM):
    """
    Freezes all base parameters of LLaMA but enables gradients for TransMLA
    projection matrices: W_UV, W_UK in Absorber and R_KV in Cruncher.
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze absorber and cruncher parameters
    for name, module in model.named_modules():
        if isinstance(module, TransMLAAbsorber):
            # absorber projection matrices
            if hasattr(module, 'W_UK'):
                module.W_UK.requires_grad = True
            if hasattr(module, 'W_UV'):
                module.W_UV.requires_grad = True
        elif isinstance(module, TransMLACruncher):
            # cruncher down projection matrix
            if hasattr(module, 'R_KV'):
                module.R_KV.requires_grad = True

class HealingTrainer:
    def __init__(self, model: LlamaForCausalLM, config: StrataKVConfig):
        self.model = model
        self.config = config
        
        # Patch the model with Strata hooks
        patch_llama_for_strata(self.model, self.config)
        prepare_for_healing(self.model)
        
    def get_trainable_parameters(self):
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def train_step(self, input_ids: torch.Tensor, prefix_len: int) -> torch.Tensor:
        """
        Executes a single curriculum training step:
        1. Forward on prefix
        2. Compresses the prefix directly into Tier 2 via forced crunching.
        3. Calculates the loss on the suffix predicting the next tokens autonomously.
        """
        self.model.train()
        
        batch_size, seq_len = input_ids.shape
        assert prefix_len < seq_len, "Prefix must be smaller than sequence length"
        assert prefix_len > 0, "Prefix length must be at least 1"
        
        prefix_ids = input_ids[:, :prefix_len]
        suffix_ids = input_ids[:, prefix_len:]
        
        # Fresh cache for forward pass
        cache = StrataKVCache(self.config)
        # Ensure it knows about the tiers
        unwrapped_model = getattr(self.model, "module", self.model)
        cache._ensure_initialized(unwrapped_model.config.num_hidden_layers - 1)
                
        # 1. Forward Prefix to populate Tier 1
        outputs_prefix = self.model(
            input_ids=prefix_ids,
            past_key_values=cache,
            use_cache=True,
            output_attentions=False
        )
        
        # --- FIX: Instantly drop the massive unused logits tensor from VRAM ---
        del outputs_prefix
        
        # Organic cache spilling from Tier 1 to Tier 2 has already occurred during the prefix pass.
        # The 900+ spilled tokens in Tier 2 have their autograd computationally linked to the Cruncher's R_KV matrix.
        
        # 3. Forward Suffix to calculate loss
        # Suffix predicting next tokens given past_key_values (now solely residing in Tier 2)
        position_ids = torch.arange(prefix_len, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        outputs_suffix = self.model(
            input_ids=suffix_ids,
            past_key_values=cache,
            position_ids=position_ids,
            use_cache=True,
        )
        
        logits = outputs_suffix.logits # [batch, suffix_len, vocab_size]
        
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = suffix_ids[..., 1:].contiguous()
        
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return loss
