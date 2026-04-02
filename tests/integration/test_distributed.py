import os
import sys
import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import LlamaConfig, LlamaForCausalLM

# Add project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.core.config import StrataKVConfig
from src.models.llama.tier3_phase5_healing import Tier3HealingTrainer

def create_dummy_matrices(path: str, num_layers: int, head_dim: int, num_kv_heads: int):
    matrices = {}
    for i in range(num_layers):
        matrices[i] = {
            # Dummy TransMLA weights
            "W_UK": torch.randn(head_dim * num_kv_heads, 32),
            "W_UV": torch.randn(head_dim * num_kv_heads, 32),
            "R_KV": torch.randn(32 * 2, head_dim * num_kv_heads),
            # Sonic weight placeholder
            "sonic_cruncher_state": {
                "medoid_proj.weight": torch.randn(32, 32 * num_kv_heads),
                "medoid_proj.bias": torch.randn(32)
            }
        }
    torch.save(matrices, path)

def main():
    accelerator = Accelerator()
    
    # 1. Setup Dummy Model and Config
    model_config = LlamaConfig(
        vocab_size=32000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        hidden_act="silu",
        max_position_embeddings=4096,
        rope_theta=10000.0,
    )
    
    matrices_path = f"/tmp/dummy_matrices_{accelerator.process_index}.pt"
    
    # Only have process 0 create the dummy file, then wait
    if accelerator.is_main_process:
        create_dummy_matrices(matrices_path, num_layers=2, head_dim=16, num_kv_heads=2)
    accelerator.wait_for_everyone()
    
    model = LlamaForCausalLM(model_config)
    model.to(accelerator.device) # Optional if we let accelerate handle it!
    
    skv_config = StrataKVConfig(
        tier0_size=4,
        tier1_size=8,
        tier2_size=16,
        enable_tier0=True,
        enable_tier1=True,
        enable_tier2=True,
        enable_tier3=True,
        transmla_matrices_path=matrices_path, # Wait, for a shared file, all processes need to read it!
        head_dim=16,
        num_kv_heads=2
    )
    
    # Re-sync matrices path if necessary since each process appended its rank. Actually let process 0 just make one shared file:
    shared_matrices_path = "/tmp/dummy_matrices_shared.pt"
    if accelerator.is_main_process:
        create_dummy_matrices(shared_matrices_path, num_layers=2, head_dim=16, num_kv_heads=2)
    accelerator.wait_for_everyone()
    skv_config.transmla_matrices_path = shared_matrices_path
    
    # 2. Setup Trainer and Accelerate
    trainer = Tier3HealingTrainer(model, skv_config, alpha_recon=1.0)
    trainable_params = trainer.get_trainable_parameters()
    
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-3)
    
    # Standard dummy dataloader
    # Generate same random inputs across instances but we will chunk them? 
    # Dataloader handles it in full distributed pipeline, but here we just manually pass equal tensors
    torch.manual_seed(42 + accelerator.process_index)
    dummy_input_ids = torch.randint(0, 32000, (1, 64))
    
    # We will pass dummy_input_ids safely through wrapper but don't need dataloader for manual tests
    trainer.model, optimizer = accelerator.prepare(trainer.model, optimizer)
    
    # 3. Distributed Train Step
    # Each process has different generated inputs
    loss, loss_dict = trainer.train_step(
        dummy_input_ids.to(accelerator.device), 
        prefix_len=32,
        k_budget=2,
        abit_threshold=0.5
    )
    
    # Verify backward pass doesn't hang
    accelerator.backward(loss)
    accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    
    # If we reached here without hanging DDP ring, success!
    if accelerator.is_main_process:
        print("[SUCCESS] Distributed pass completed cleanly.")
        
    # Cleanup
    if accelerator.is_main_process:
        if os.path.exists(shared_matrices_path):
            os.remove(shared_matrices_path)

if __name__ == "__main__":
    main()
