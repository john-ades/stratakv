import os
import sys
import torch

# Add project root to python path to import our modules
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("Please install transformers to run this script.")
    sys.exit(1)

def main():
    print("--- StrataKV-Torch Baseline Runner for Qwen 3 ---")
    # Configure our StrataKV Cache sizes
    # Tier 0 (sink/system prompt): keeps first 4 tokens permanently
    # Tier 1 (sliding window L1): keeps most recent 2048 tokens
    config = StrataKVConfig(
        tier0_size=4,
        tier1_size=2048,
        enable_tier0=True, # Active
        enable_tier1=True, # Active
        enable_tier2=False, # We drop everything else
        enable_tier3=False
    )
    
    # Initialize cache manager
    cache = StrataKVCache(config)
    print(f"StrataKV Configuration: T0={config.tier0_size}, T1={config.tier1_size}")
    
    # Placeholder for Qwen 3 model ID. Update if the exact name differs.
    model_id = "Qwen/Qwen3-1.7B"
    
    print(f"Loading tokenizer and model ({model_id})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            dtype=torch.float16, 
            device_map="auto",
            trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model {model_id} - check huggingface login & access.")
        print(f"Error: {e}")
        return

    prompt = "Hello, I am a tiered cascaded caching architecture designed to compress contexts for infinite generation."
    
    # Qwen typically uses specific template formats, but a simple encode is sufficient for baseline testing
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    except Exception:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    print(f"Starting Generation with Input token count = {inputs.input_ids.shape[1]}")
    
    with torch.no_grad():
        # A single forward pass with the custom caching layer injected
        outputs = model(
            input_ids=inputs.input_ids,
            past_key_values=cache,
            use_cache=True
        )
        
    print(f"Generated Logits shape: {outputs.logits.shape}")
    
    # Let's peek into Layer 0's cache footprints
    if len(cache._tier0_sinks) > 0:
        t0_len = cache._tier0_sinks[0].seq_len if cache._tier0_sinks[0] else 0
        t1_len = cache._tier1_recents[0].seq_len if cache._tier1_recents[0] else 0
        print(f"Layer 0 Memory Footprint => Tier 0 Tokens: {t0_len} | Tier 1 Tokens: {t1_len}")
    
    print("Baseline test complete. Tier 0 and Tier 1 successfully hooked for Qwen 3.")

if __name__ == "__main__":
    main()
