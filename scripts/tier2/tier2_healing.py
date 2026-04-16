import os
import sys
import torch
import typer
from rich.console import Console
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.core.config import StrataKVConfig, DEFAULT_MODEL_ID
from src.models.llama.tier2_phase5_healing import HealingTrainer
from src.compression.transmla import TransMLACruncher, TransMLAAbsorber


app = typer.Typer(help="Tier 2 Phase 5: TransMLA Curriculum Healing Finetuning")
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
        "HuggingFaceFW/fineweb-edu", 
        "--dataset", "-d",
        help="Dataset name for training"
    ),
    dataset_config: str = typer.Option(
        "sample-10BT", 
        "--dataset-config", "-c",
        help="Dataset config name"
    ),
    matrices_path: str = typer.Option(
        "outputs/llama_transmla_base.pt", 
        "--matrices-path", "-p",
        help="Path to the extracted untrained Phase 1 TransMLA matrices"
    ),
    output_path: str = typer.Option(
        "outputs/llama_transmla_healed.pt", 
        "--output-path", "-o",
        help="Path to save the fine-tuned 'Healed' TransMLA matrices"
    ),
    batch_size: int = typer.Option(1, "--batch-size", "-b", help="Batch size per forward pass"),
    lr: float = typer.Option(2e-5, "--learning-rate", "-lr", help="Learning rate for AdamW"),
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of epochs to train"),
    max_steps: int = typer.Option(500, "--max-steps", "-s", help="Max training iterations"),
    seq_len: int = typer.Option(4096, "--seq-len", "-l", help="Total sequence length"),
    prefix_len: int = typer.Option(2048, "--prefix-len", "-pl", help="Prefix length to force into Tier 1/2"),
    tier1_size: int = typer.Option(1024, "--tier1-size", "-t1", help="Tier 1 sliding window size"),
    device: str = typer.Option("auto", "--device", "-dev", help="Device to use (auto, cpu, cuda, mps)"),
):
    """
    Curriculum training script to heal the boundary between the L1 cache 
    and the aggressively compressed L2 TransMLA cache.
    """
    console.print(f"[bold green]Starting Phase 5 Healing for {model_id}[/bold green]")
    
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

    # Instantiate Tier 2 active config
    config = StrataKVConfig(
        tier0_size=4,
        tier1_size=tier1_size, # Passed from CLI
        enable_tier0=True,
        enable_tier1=True,
        enable_tier2=True,
        enable_tier3=False,
        transmla_matrices_path=matrices_path,
        head_dim=model_head_dim,
        num_kv_heads=model_num_kv_heads
    )
    
    # Enforce eviction happening during forward pass
    if prefix_len <= config.tier1_size:
        console.print(f"[bold yellow]Warning: prefix_len ({prefix_len}) should be > tier1_size ({config.tier1_size}) to force crunching to Tier 2![/bold yellow]")
    
    console.print("Initializing HealingTrainer (Freezing Base, Active TransMLA parameters)...")
    trainer = HealingTrainer(base_model, config)
    
    trainable_params = trainer.get_trainable_parameters()
    console.print(f"[bold green]Found {len(trainable_params)} trainable matrices across all layers.[/bold green]")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    
    console.print(f"Loading {dataset_name} (streaming mode)...")
    
    if dataset_name == "dummy":
        console.print("[bold cyan]Generating dummy tensor data for instantaneous smoke testing![/bold cyan]")
        # Pre-generate random token batches matching vocab size (128256 for LLaMA 3)
        dataloader = [torch.randint(0, 10000, (batch_size, seq_len)) for _ in range(max_steps)]
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
    
    console.print("\n[bold green]=== Starting Training Loop ===[/bold green]")
    
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
            
            try:
                # 1. Forward pass mapping (Prefix / Suffix)
                loss = trainer.train_step(batch_input_ids, prefix_len=prefix_len)
                
                # 2. Backprop
                accelerator.backward(loss)
                optimizer.step()
                
                epoch_bar.update(1)
                epoch_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
                
                global_step += 1
            except Exception as e:
                console.print(f"[bold red]Training step failed: {str(e)}[/bold red]")
                
        if global_step >= max_steps:
            break
            
    epoch_bar.close()
    
    # Extract matrices and save them
    console.print("\n[bold green]Training Complete. Extracting updated matrices...[/bold green]")
    
    # Load original structure to overwrite with trained matrices
    matrices_dict = torch.load(matrices_path, map_location="cpu", weights_only=True)
    
    unwrapped_model = accelerator.unwrap_model(trainer.model)
    for name, module in unwrapped_model.named_modules():
        if isinstance(module, TransMLAAbsorber) or isinstance(module, TransMLACruncher):
            l_idx = module.layer_idx
            if hasattr(module, "W_UK"):
                matrices_dict[l_idx]["W_UK"] = module.W_UK.detach().cpu()
            if hasattr(module, "W_UV"):
                matrices_dict[l_idx]["W_UV"] = module.W_UV.detach().cpu()
            if hasattr(module, "R_KV") and isinstance(module, TransMLACruncher):
                matrices_dict[l_idx]["R_KV"] = module.R_KV.detach().cpu()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(matrices_dict, output_path)
    console.print(f"[bold blue]Successfully saved Healed Matrices to {output_path}[/bold blue]")

if __name__ == "__main__":
    app()
