import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaForCausalLM

from ...cache_manager import StrataKVCache
from ...core.config import StrataKVConfig
from ...compression.transmla import TransMLAAbsorber, TransMLACruncher
from ...compression.sonic import SonicCruncher
from .modeling_llama import patch_llama_for_strata

def prepare_for_healing(model: LlamaForCausalLM):
    """
    Freezes all base parameters of LLaMA and Tier 2 TransMLA matrices.
    Enables gradients solely for Tier 3 SonicCruncher projection matrices.
    """
    # 1. Freeze everything
    for param in model.parameters():
        param.requires_grad = False
        
    # 2. Unfreeze SonicCruncher parameters
    for name, module in model.named_modules():
        if isinstance(module, SonicCruncher):
            for param in module.parameters():
                param.requires_grad = True

class Tier3HealingTrainer:
    def __init__(self, model: LlamaForCausalLM, config: StrataKVConfig, alpha_recon: float = 1.0):
        self.model = model
        self.config = config
        self.alpha_recon = alpha_recon
        
        # Patch the model with Strata hooks
        patch_llama_for_strata(self.model, self.config)
        prepare_for_healing(self.model)
        
    def get_trainable_parameters(self):
        params = []
        for param in self.model.parameters():
            if param.requires_grad:
                params.append(param)
        return params

    def train_step(self, input_ids: torch.Tensor, prefix_len: int, k_budget: int, abit_threshold: float) -> tuple[torch.Tensor, dict]:
        """
        Executes a curriculum training step for Tier 3:
        1. Teacher Pass: Standard Dense LLaMA forward pass to obtain teacher logits.
        2. Applies dynamic budgeting overrides.
        3. Student Prefix: Spills tokens naturally into T1 -> T2 -> T3.
        4. Student Suffix: Calculates outputs and derives distillation and reconstruction losses.
        """
        batch_size, seq_len = input_ids.shape
        assert prefix_len < seq_len, "Prefix must be smaller than sequence length"
        assert prefix_len > 0, "Prefix length must be at least 1"
        
        # 1. Teacher Pass (Dense Uncompressed)
        self.model.eval()
        self.config.enable_tier2 = False
        self.config.enable_tier3 = False
        
        teacher_cache = StrataKVCache(self.config)
        with torch.no_grad():
            outputs_teacher = self.model(
                input_ids=input_ids,
                past_key_values=teacher_cache,
                use_cache=True,
                output_attentions=False
            )
            teacher_logits = outputs_teacher.logits[:, prefix_len:-1, :].contiguous()
            
        # --- FIX: Free 2GB+ of teacher outputs/cache ---
        del outputs_teacher
        del teacher_cache
            
        # 2. Prepare Student Pass
        self.config.enable_tier2 = True
        self.config.enable_tier3 = True
        self.config.tier3_k = k_budget
        self.config.abit_initial_threshold = abit_threshold
        self.model.train()
        
        # Clear any accumulated reconstruction losses
        for module in self.model.modules():
            if hasattr(module, "recon_loss"):
                module.recon_loss = None

        student_cache = StrataKVCache(self.config)
        unwrapped_model = getattr(self.model, "module", self.model)
        student_cache._ensure_initialized(unwrapped_model.config.num_hidden_layers - 1)
        
        prefix_ids = input_ids[:, :prefix_len]
        suffix_ids = input_ids[:, prefix_len:]
        
        # 3. Forward Prefix (Populates T1 -> T2 -> T3)
        outputs_prefix = self.model(
            input_ids=prefix_ids,
            past_key_values=student_cache,
            use_cache=True,
            output_attentions=False
        )
        
        # --- FIX: Instantly drop the massive unused prefix logits tensor ---
        del outputs_prefix
        
        # 4. Forward Suffix
        position_ids = torch.arange(prefix_len, seq_len, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        outputs_suffix = self.model(
            input_ids=suffix_ids,
            past_key_values=student_cache,
            position_ids=position_ids,
            use_cache=True,
        )
        
        student_logits_full = outputs_suffix.logits
        student_logits = student_logits_full[:, :-1, :].contiguous()
        
        # Calculate KD Loss
        loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
        log_student = nn.functional.log_softmax(student_logits, dim=-1)
        log_teacher = nn.functional.log_softmax(teacher_logits, dim=-1)
        
        l_kd = loss_fct(log_student.view(-1, log_student.size(-1)), log_teacher.view(-1, log_teacher.size(-1)))
        
        # Calculate Recon Loss
        l_recon = 0.0
        recon_count = 0
        for module in self.model.modules():
            if hasattr(module, "recon_loss") and module.recon_loss is not None:
                l_recon += module.recon_loss
                recon_count += 1
                module.recon_loss = None
                
        if recon_count > 0:
            l_recon = l_recon / recon_count
            
        total_loss = l_kd + self.alpha_recon * l_recon
        
        loss_dict = {
            "Total": total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
            "L_KD": l_kd.item(),
            "L_Recon": l_recon.item() if isinstance(l_recon, torch.Tensor) else l_recon
        }
        
        return total_loss, loss_dict
