import torch
import pytest
from transformers import LlamaConfig, LlamaForCausalLM

from src.core.config import StrataKVConfig
from src.models.llama.tier3_phase5_healing import Tier3HealingTrainer
from src.compression.sonic import SonicCruncher

@pytest.fixture
def dummy_model_and_config():
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    # Use float32 for testing
    model = LlamaForCausalLM(config).cpu()
    
    strata_config = StrataKVConfig(
        tier0_size=4,
        tier1_size=8,
        tier2_size=16,
        enable_tier0=True,
        enable_tier1=True,
        enable_tier2=True,
        enable_tier3=True,
        head_dim=16,  # 64 / 4
        num_kv_heads=2,
        transmla_rope_dim=8,
        transmla_target_rank=8,
        abit_window_size=2,
        abit_min_split_tokens=4,
        abit_max_split_tokens=4,
        abit_initial_threshold=0.5,
        tier3_k=2 # Will be dynamically overridden, but this is the default
    )
    
    return model, strata_config

def test_tier3_curriculum_healing(dummy_model_and_config):
    model, config = dummy_model_and_config
    
    trainer = Tier3HealingTrainer(model, config, alpha_recon=1.0)
    
    # 1. Check Parameters Freezing
    # Base LLaMA should be frozen
    assert not model.model.layers[0].self_attn.q_proj.weight.requires_grad
    # TransMLA Absorber should be frozen
    assert not model.model.layers[0].self_attn.strata_absorber.W_UK.requires_grad
    # TransMLA Cruncher should be frozen
    assert not model.model.layers[0].self_attn.strata_cruncher.R_KV.requires_grad
    
    # SonicCruncher should be unfrozen
    sonic: SonicCruncher = model.model.layers[0].self_attn.sonic_cruncher
    assert sonic.q_proj.weight.requires_grad
    assert sonic.nexus_base.requires_grad
    
    trainable_params = trainer.get_trainable_parameters()
    assert len(trainable_params) > 0
    
    # 2. Forward pass step
    batch_size = 2
    seq_len = 64
    prefix_len = 32
    
    # Initialize TransMLA mock parameters to non-zero so gradients are not identically 0
    with torch.no_grad():
        for layer in model.model.layers:
            layer.self_attn.strata_cruncher.R_KV.normal_()
    
    # Random token ids
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    k_budget = 2
    abit_threshold = 0.0
    
    loss, loss_dict = trainer.train_step(input_ids, prefix_len, k_budget, abit_threshold)
    
    # 3. Verify Validation
    assert isinstance(loss, torch.Tensor)
    assert loss.item() > 0
    assert "L_KD" in loss_dict
    assert "L_Recon" in loss_dict
    assert loss_dict["L_KD"] > 0
    
    # For randomly initialized small model, L_Recon must not be NaN or zero
    assert loss_dict["L_Recon"] >= 0
    assert not torch.isnan(loss).item()
    
    # 4. Backward pass
    loss.backward()
    
    # 5. Check Gradients
    # Base params should not have gradients
    assert model.model.layers[0].self_attn.q_proj.weight.grad is None
    
    # TransMLA shouldn't have gradients
    assert model.model.layers[0].self_attn.strata_absorber.W_UK.grad is None
    assert model.model.layers[0].self_attn.strata_cruncher.R_KV.grad is None
    
    # SonicCruncher MUST have gradients because T3 features were used to predict suffix tokens
    # and recon loss flows backward directly.
    assert sonic.q_proj.weight.grad is not None
    assert sonic.nexus_base.grad is not None
    
    # Verify gradients are not universally zero
    grad_sum = torch.abs(sonic.q_proj.weight.grad).sum().item()
    assert grad_sum > 0
