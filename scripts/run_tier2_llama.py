import os
import sys
import torch
import typer
from rich.console import Console
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add project root to python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
from src.models.llama.modeling_llama import patch_llama_for_strata

app = typer.Typer(help="Tier 2 Phase 6: E2E LLaMA TransMLA Inference")
console = Console()

@app.command()
def generate(
    model_id: str = typer.Option(
        "meta-llama/Llama-3.2-1B-Instruct",
        "--model-id", "-m",
        help="HuggingFace Model ID to evaluate"
    ),
    matrices_path: str = typer.Option(
        "outputs/llama_transmla_healed.pt", 
        "--matrices-path", "-p",
        help="Path to the fine-tuned 'Healed' TransMLA matrices"
    ),
    tier1_size: int = typer.Option(
        256, 
        "--tier1-size", "-t1",
        help="Size of L1 cache sliding window before evicting to Tier 2"
    ),
    max_new_tokens: int = typer.Option(
        250, 
        "--max-new-tokens", "-n",
        help="Number of tokens to generate autoregressively"
    )
):
    """
    An end-to-end generation and evaluation script utilizing the fully trained StrataKV Tier 2 cache.
    Proof of Phase 4 (Hybrid Dual-Path Attention) via Late Decompression.
    """
    console.print(f"[bold green]--- StrataKV E2E Generation Runner ---[/bold green]")
    
    if not os.path.exists(matrices_path):
        console.print(f"[bold red]Matrices file not found at {matrices_path}![/bold red]")
        sys.exit(1)
        
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    console.print(f"Loading tokenizer and model ({model_id}) onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map=device
        )
        model.eval()
        
        # Calculate dynamically
        model_head_dim = model.config.hidden_size // model.config.num_attention_heads
        model_num_kv_heads = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

        # Instantiate Active Tier 2 config
        config = StrataKVConfig(
            tier0_size=4,
            tier1_size=tier1_size,
            enable_tier0=True,
            enable_tier1=True,
            enable_tier2=True,
            enable_tier3=False,
            transmla_matrices_path=matrices_path,
            head_dim=model_head_dim,
            num_kv_heads=model_num_kv_heads
        )
        
        console.print(f"StrataKV Configuration: T0={config.tier0_size}, T1={config.tier1_size}, T2=ACTIVE")
        cache = StrataKVCache(config)
        
        console.print("Patching LLaMA model for StrataKV Hybrid Attention...")
        patch_llama_for_strata(model, config)
        
    except Exception as e:
        console.print(f"[bold red]Failed to load or patch model {model_id}.[/bold red]")
        console.print(f"Error: {e}")
        return

    # A massively long prompt that securely exceeds Tier 1 size (256 here).
    # This guarantees tokens will forcefully spill off Tier 1's sliding window and into 
    # the TransMLA Cruncher (Phase 3) into the Tier 2 Absorber.
    massive_prompt = (
        "The following is a long hypothetical text explaining the inner workings of memory systems "
        "in large language models. The problem with traditional systems is that KV caches scale "
        "linearly with the number of tokens, which bottlenecks context windows quadratically when "
        "coupled with standard attention mechanisms. To combat this, researchers use tiered cascaded "
        "architectures. Let us dive deep into the first tier. Tier zero is often called the sink tier, "
        "which keeps initial system prompt tokens permanently pinned so attention doesn't collapse. "
        "Tier one is a standard sliding window, maintaining exact L1 precision for recency bias. "
        "However, to go infinitely long, tokens must eventually be evicted. If evicted into standard "
        "RAM, decoding becomes drastically slow due to PCIe latency. This brings us to Tier 2: TransMLA. "
        "TransMLA projects Keys and Values using RoRoPE orthogonal rotation and Balanced KV PCA. "
        "By doing this, tokens seamlessly transition into a tiny, contiguous rank space inside VRAM. "
        "Since we've patched LLaMA correctly, we should now be able to retrieve these far-away concepts "
        "without hallucinating, right? The key to memory is realizing that not all tokens are created equal. "
        "Tell me what you think about this cascaded cache mechanism based on what I just wrote."
    )
    
    inputs = tokenizer(massive_prompt, return_tensors="pt").to(model.device)
    
    console.print(f"[bold yellow]Input prompt token count = {inputs.input_ids.shape[1]}[/bold yellow]")
    if inputs.input_ids.shape[1] < config.tier1_size:
        console.print(f"[bold red]Warning: Prompt length ({inputs.input_ids.shape[1]}) is less than T1 Size ({config.tier1_size}). Tokens will NOT spill to Tier 2.[/bold red]")
    else:
        console.print("[bold blue]Excellent: Prompt length exceeds T1 size. Evictions to Tier 2 guarantee'd.[/bold blue]")

    console.print("\n[bold green]Generating (Autoregressively)...[/bold green]")
    
    generated_ids = inputs.input_ids
    
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(
                input_ids=generated_ids if i == 0 else generated_ids[:, -1:],
                past_key_values=cache,
                use_cache=True
            )
            
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Print token via console for streaming effect
            console.out(tokenizer.decode(next_token[0]), end="", highlight=False)
            
            # Simple break on EOS
            if next_token[0].item() == tokenizer.eos_token_id:
                break
                
    console.print("\n\n[bold green]--- Generation Complete ---[/bold green]")
    
    # Layer 0 footprint summary
    if len(cache._tier0_sinks) > 0:
        t0_len = cache._tier0_sinks[0].seq_len if cache._tier0_sinks[0] else 0
        t1_len = cache._tier1_recents[0].seq_len if cache._tier1_recents[0] else 0
        t2_len = 0
        if config.enable_tier2 and hasattr(cache, "_tier2_compressed") and 0 in cache._tier2_compressed:
            c_kv_len = cache._tier2_compressed[0].c_kv.shape[1]
            t2_len = c_kv_len
            
        console.print(f"\n[bold magenta]Layer 0 Memory Footprint[/bold magenta]")
        console.print(f"Tier 0 (Uncompressed Sink):     {t0_len} tokens")
        console.print(f"Tier 1 (Uncompressed Sliding):  {t1_len} tokens")
        console.print(f"Tier 2 (TransMLA Compressed):   {t2_len} tokens")

    console.print("\n[bold blue]End of Test[/bold blue]")

if __name__ == "__main__":
    app()
