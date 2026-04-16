import os
import sys
import torch
import typer
import random
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, interleave_datasets, IterableDataset
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.core.config import StrataKVConfig, DEFAULT_MODEL_ID
from src.models.llama.tier3_phase5_healing import Tier3HealingTrainer
from src.compression.sonic import SonicCruncher

app = typer.Typer(help="Tier 3 Phase 5: SONIC Curriculum Healing Finetuning")
console = Console()

def collate_fn(batch, tokenizer, max_length):
    texts = [item["text"] for item in batch if len(item["text"].strip()) > 0]
    if not texts:
        return None
    encodings = tokenizer(
        texts, 
        truncation=True, 
        max_length=max_length, 
        padding="max_length", 
        return_tensors="pt"
    )
    return encodings.input_ids

@app.command()
def heal(
    model_id: str = typer.Option(
        DEFAULT_MODEL_ID,
        "--model-id", "-m",
        help="HuggingFace Model ID to heal"
    ),
    dataset_name: str = typer.Option(
        "mixture", 
        "--dataset", "-d",
        help="Dataset name for training"
    ),
    dataset_config: str = typer.Option(
        "sample-10BT", 
        "--dataset-config", "-c",
        help="Dataset config name"
    ),
    matrices_path: str = typer.Option(
        "outputs/llama_transmla_healed.pt", 
        "--matrices-path", "-p",
        help="Path to the TransMLA matrices (Tier 2)"
    ),
    output_path: str = typer.Option(
        "outputs/llama_sonic_healed.pt", 
        "--output-path", "-o",
        help="Path to save the combined Healed TransMLA & Tier 3 SONIC matrices"
    ),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size per forward pass"),
    lr: float = typer.Option(5e-5, "--learning-rate", "-lr", help="Learning rate for AdamW"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of epochs to train"),
    max_steps: int = typer.Option(500, "--max-steps", "-s", help="Max training iterations"),
    seq_len: int = typer.Option(4096, "--seq-len", "-l", help="Total sequence length"),
    prefix_len: int = typer.Option(2048, "--prefix-len", "-pl", help="Prefix length to force into Tiers 1/2/3"),
    tier1_size: int = typer.Option(256, "--tier1-size", "-t1", help="Tier 1 sliding window size"),
    tier2_size: int = typer.Option(1024, "--tier2-size", "-t2", help="Tier 2 latent capacity"),
    alpha_recon: float = typer.Option(1.0, "--alpha-recon", "-ar", help="Weight for reconstruction loss"),
    device: str = typer.Option("auto", "--device", "-dev", help="Device to use (auto, cpu, cuda, mps)"),
):
    """
    Curriculum training script to distill uncompressed context into highly compressed 
    SONIC Tier 3 Nexus tokens, optimizing both KD and Reconstruction Loss.
    """
    console.print(f"[bold green]Starting Phase 5 Tier 3 SONIC Healing for {model_id}[/bold green]")
    
    accelerator = Accelerator()
    
    if not os.path.exists(matrices_path):
        console.print(f"[bold red]Matrices file not found at {matrices_path}![/bold red]")
        sys.exit(1)
        
    device = accelerator.device
    console.print(f"[bold yellow]Using device: {device}[/bold yellow]")
    
    console.print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    
    # Calculate head_dim and num_kv_heads dynamically
    model_head_dim = base_model.config.hidden_size // base_model.config.num_attention_heads
    model_num_kv_heads = getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads)

    # Instantiate StrataKVConfig
    config = StrataKVConfig(
        tier0_size=4,
        tier1_size=tier1_size,
        tier2_size=tier2_size,
        enable_tier0=True,
        enable_tier1=True,
        enable_tier2=True,
        enable_tier3=True, # Active
        transmla_matrices_path=matrices_path,
        head_dim=model_head_dim,
        num_kv_heads=model_num_kv_heads
    )
    
    console.print("Initializing Tier3HealingTrainer (Freezing Base + TransMLA, training SONIC)...")
    trainer = Tier3HealingTrainer(base_model, config, alpha_recon=alpha_recon)
    
    trainable_params = trainer.get_trainable_parameters()
    console.print(f"[bold green]Found {len(trainable_params)} trainable SONIC tensors across all layers.[/bold green]")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    console.print(f"Loading {dataset_name} (streaming mode)...")
    
    if dataset_name == "dummy":
        console.print("[bold cyan]Generating dummy tensor data for instantaneous smoke testing![/bold cyan]")
        dataloader = [torch.randint(0, 10000, (batch_size, seq_len)) for _ in range(max_steps)]
    elif dataset_name == "mixture":
        console.print("[bold cyan]Loading SONIC mixture (SOC, UltraChat, TopiOCQA)...[/bold cyan]")
        
        def format_soc(example):
            text = ""
            for part in example.get("chat_parts", []):
                sender = part.get("sender", "User")
                msgs = part.get("messages", [])
                text += f"{sender}: " + " ".join(msgs) + "\n"
            return {"text": text}
            
        def format_ultrachat(example):
            text = ""
            for msg in example.get("messages", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                text += f"{role}: {content}\n"
            return {"text": text}
            
        def format_coqa(example):
            text = "Context: " + example.get("story", "") + "\n"
            questions = example.get("questions", [])
            answers = example.get("answers", {}).get("input_text", [])
            for q, a in zip(questions, answers):
                text += f"User: {q}\nExpert: {a}\n"
            return {"text": text}
            
        ds_soc = load_dataset("marcodsn/SOC-2508", split="train", streaming=True).map(format_soc)
        ds_soc = ds_soc.select_columns(["text"])
        
        ds_ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).map(format_ultrachat)
        ds_ultra = ds_ultra.select_columns(["text"])
        
        ds_coqa = load_dataset("coqa", split="train", streaming=True).map(format_coqa)
        ds_coqa = ds_coqa.select_columns(["text"])
        
        dataset = interleave_datasets([ds_soc, ds_ultra, ds_coqa], probabilities=[0.34, 0.33, 0.33])
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=lambda b: collate_fn(b, tokenizer, seq_len)
        )
    else:
        dataset = load_dataset(dataset_name, name=dataset_config, split="train", streaming=True)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            collate_fn=lambda b: collate_fn(b, tokenizer, seq_len)
        )
        
    trainer.model, optimizer, dataloader = accelerator.prepare(
        trainer.model, optimizer, dataloader
    )
    
    console.print("\n[bold green]=== Starting Curriculum Distillation ===[/bold green]")
    
    global_step = 0
    epoch_bar = tqdm(total=max_steps, desc="Healing Steps")
    
    for epoch in range(epochs):
        for batch_input_ids in dataloader:
            if batch_input_ids is None:
                continue
            
            if global_step >= max_steps:
                break
            
            batch_input_ids = batch_input_ids.to(device)
            
            optimizer.zero_grad()
            
            # --- Dynamic Budgeting Overrides ---
            # Randomly sample K to force dimension robustness
            k_budget = random.choice([1, 2, 4])
            # Randomly fluctuate clustering threshold to train on tiny and massive sequence spans
            abit_threshold = random.uniform(0.3, 0.7)
            
            try:
                loss, loss_dict = trainer.train_step(
                    batch_input_ids, 
                    prefix_len=prefix_len,
                    k_budget=k_budget,
                    abit_threshold=abit_threshold
                )
                
                accelerator.backward(loss)
                # Gradient clipping to stabilize Transformer block training
                accelerator.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                epoch_bar.update(1)
                desc = f"Loss: {loss.item():.3f} | L_KD: {loss_dict['L_KD']:.3f} | L_Rc: {loss_dict['L_Recon']:.3f} | K: {k_budget} | Th: {abit_threshold:.2f}"
                epoch_bar.set_postfix_str(desc)
                
                global_step += 1
            except Exception as e:
                console.print(f"\\n[bold red]Training step failed: {str(e)}[/bold red]")
                
        if global_step >= max_steps:
            break
            
    epoch_bar.close()
    
    console.print("\n[bold green]Training Complete. Extracting Tier 3 matrices...[/bold green]")
    
    # Load original structure to overwrite and add Sonic weights
    matrices_dict = torch.load(matrices_path, map_location="cpu", weights_only=True)
    
    unwrapped_model = accelerator.unwrap_model(trainer.model)
    for name, module in unwrapped_model.named_modules():
        if module.__class__.__name__ == "LlamaAttention":
            l_idx = module.layer_idx
            if hasattr(module, "sonic_cruncher"):
                # Make sure the layer dict exists
                if l_idx not in matrices_dict:
                    matrices_dict[l_idx] = {}
                    
                # Convert state_dict to CPU tensors
                c_state = module.sonic_cruncher.state_dict()
                cpu_state = {k: v.detach().cpu() for k, v in c_state.items()}
                matrices_dict[l_idx]["sonic_cruncher_state"] = cpu_state

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Needs weights_only=False if we save arbitrary dicts for state dict... 
    # But dictionary of tensors perfectly satisfies weights_only=True.
    torch.save(matrices_dict, output_path)
    console.print(f"[bold blue]Successfully saved Tier 3 SONIC Healed Matrices to {output_path}[/bold blue]")

if __name__ == "__main__":
    app()
