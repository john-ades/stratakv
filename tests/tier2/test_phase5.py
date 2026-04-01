import torch
import torch.nn as nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM
import pytest

from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
from src.models.llama.tier2_phase5_healing import HealingTrainer, prepare_for_healing

@pytest.fixture
def dummy_llama():
    config = LlamaConfig(
        vocab_size=100,
        hidden_size=64, # head_dim = hidden_size / num_heads = 64 / 4 = 16
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024
    )
    model = LlamaForCausalLM(config)
    return model

@pytest.fixture
def strata_config():
    return StrataKVConfig(
        num_kv_heads=2,
        head_dim=16,
        
        # Tiering
        enable_tier0=True,
        tier0_size=4,
        
        enable_tier1=True,
        tier1_size=8,
        
        enable_tier2=True,
        tier2_size=100,
        
        # Compression
        transmla_rope_dim=8,
        transmla_target_rank=12,
        transmla_matrices_path=None
    )

def test_prepare_for_healing(dummy_llama, strata_config):
    trainer = HealingTrainer(dummy_llama, strata_config)
    
    # Check that base parameters are frozen
    assert dummy_llama.model.embed_tokens.weight.requires_grad == False
    assert dummy_llama.model.layers[0].self_attn.q_proj.weight.requires_grad == False
    
    # Check that projection parameters are trainable
    trainable_params = trainer.get_trainable_parameters()
    
    found_W_UK = False
    found_W_UV = False
    found_R_KV = False
    
    for param in trainable_params:
        if param.shape == (8, 12): # nope_dim (16-8) = 8, target_rank = 12 -> W_UK is (8, 12)
            found_W_UK = True
        elif param.shape == (12, 16): # W_UV is (12, 16)
            found_W_UV = True
        elif param.shape == (48, 12): # in_features = 8*2 + 16*2 = 48, target_rank = 12 -> R_KV is (48, 12)
            found_R_KV = True
            
    assert found_W_UK, "W_UK was not found as a trainable parameter"
    assert found_W_UV, "W_UV was not found as a trainable parameter"
    assert found_R_KV, "R_KV was not found as a trainable parameter"


def test_healing_training_step(dummy_llama, strata_config):
    trainer = HealingTrainer(dummy_llama, strata_config)
    
    # Initialize TransMLA mock parameters so c_kv isn't exactly zero
    with torch.no_grad():
        for layer in dummy_llama.model.layers:
            layer.self_attn.strata_absorber.W_UK.normal_()
            layer.self_attn.strata_absorber.W_UV.normal_()
            layer.self_attn.strata_cruncher.R_KV.normal_()
            
    # Create optimizer
    optimizer = torch.optim.Adam(trainer.get_trainable_parameters(), lr=1e-3)
    
    # Dummy input
    input_ids = torch.randint(0, 100, (1, 25))
    prefix_len = 15
    
    # Run training step
    loss = trainer.train_step(input_ids, prefix_len)
    
    # Loss should be non-zero and valid
    assert loss is not None
    assert loss.item() > 0
    
    loss.backward()
    optimizer.step()
    
    # Validate gradients flowed into Absorber & Cruncher
    for name, module in dummy_llama.named_modules():
        if module.__class__.__name__ == "TransMLAAbsorber":
            assert module.W_UK.grad is not None
            assert module.W_UV.grad is not None
        elif module.__class__.__name__ == "TransMLACruncher":
            assert module.R_KV.grad is not None
