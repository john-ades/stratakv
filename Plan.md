To successfully deploy your first full-scale, multi-GPU remote training and benchmarking experiment, you need to step away from isolated smoke-testing scripts and build a **Unified Orchestration Pipeline**. 

Running an experiment on a remote cluster via PyTorch Distributed Data Parallel (DDP) combined with benchmarking 128k contexts introduces four critical challenges that your local tests didn't account for:

1. **Multi-GPU File Corruption:** During distributed runs, all GPUs execute the script simultaneously. If 8 GPUs attempt to save the TransMLA `.pt` weights dictionary at the exact same millisecond, the file will corrupt. You must enforce thread barriers (`accelerator.wait_for_everyone()`).
2. **Dataset Schema Mismatch:** `McGill-NLP/TopiOCQA` has a uniquely nested schema (`Gold_passage` -> `text`). We must explicitly parse this to map its structure into a continuous streaming layout alongside `SOC-2508` and `UltraChat`.
3. **CUDA Graph Bleed:** Training Phase 2 and Phase 3 sequentially in the same script on PyTorch DDP will cause computation graph collisions. You must explicitly drop the trainer, clear the CUDA cache, and load a fresh model between phases.
4. **128k Context Evaluation OOMs (The `generate()` Trap):** Benchmarks like NeedleBench and RULER evaluate contexts up to 128k tokens. Standard Hugging Face `model.generate()` will pass all 128k tokens into the attention layer in a single shot, causing an instant $128k \times 128k$ eager attention VRAM spike (OOM). We must wrap the model with a **Chunked Prefill Evaluator**.

Here is an example pipeline script that addresses all the above issues.

### Create `scripts/run_experiment.py`

```python
import os
import sys
import types
import torch
import gc
from accelerate import Accelerator
from datasets import load_dataset, interleave_datasets
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from rich.console import Console

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.core.config import StrataKVConfig
from src.cache_manager import StrataKVCache
from src.models.llama.tier2_phase1_extraction import run_offline_calibration
from src.models.llama.tier2_phase5_healing import HealingTrainer as Tier2Trainer
from src.models.llama.tier3_phase5_healing import Tier3HealingTrainer as Tier3Trainer
from src.models.llama.modeling_llama import patch_llama_for_strata

console = Console()

# ==========================================
# 1. Dataset Mixture Configuration
# ==========================================
def build_mixed_dataloader(tokenizer, seq_len: int, batch_size: int):
    # Format SOC
    def format_soc(example):
        text = ""
        for part in example.get("chat_parts", []):
            text += f"{part.get('sender', 'User')}: {' '.join(part.get('messages', []))}\n"
        return {"text": text}
        
    # Format UltraChat
    def format_ultrachat(example):
        text = ""
        for msg in example.get("messages", []):
            text += f"{msg.get('role', 'user')}: {msg.get('content', '')}\n"
        return {"text": text}
        
    # Format TopiOCQA (Targeting specific nested schema)
    def format_topiocqa(example):
        context = example.get("Gold_passage", {}).get("text", "") if example.get("Gold_passage") else ""
        text = f"Context: {context}\n"
        text += f"User: {example.get('Question', '')}\n"
        text += f"Expert: {example.get('Answer', '')}\n"
        return {"text": text}
        
    ds_soc = load_dataset("marcodsn/SOC-2508", split="train", streaming=True).map(format_soc).select_columns(["text"])
    ds_ultra = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True).map(format_ultrachat).select_columns(["text"])
    ds_topiocqa = load_dataset("McGill-NLP/TopiOCQA", split="train", streaming=True).map(format_topiocqa).select_columns(["text"])
    
    # Mix the datasets dynamically
    dataset = interleave_datasets([ds_soc, ds_ultra, ds_topiocqa], probabilities=[0.34, 0.33, 0.33])
    
    def collate_fn(batch):
        texts = [item["text"] for item in batch if len(item["text"].strip()) > 0]
        if not texts: return None
        encodings = tokenizer(texts, truncation=True, max_length=seq_len, padding="max_length", return_tensors="pt")
        return encodings.input_ids
        
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

# ==========================================
# 2. Evaluation / Benchmark Monkey-Patch
# ==========================================
def enable_benchmark_mode(model, config: StrataKVConfig, max_chunk_size=2048):
    """
    Monkey-patches model.generate() so NeedleBench, RULER, and MTBench101 
    can evaluate StrataKV without OOMing on massive context prefilling (128k+ tokens).
    """
    original_generate = model.generate
    
    def custom_generate(self, input_ids, **kwargs):
        cache = StrataKVCache(config)
        seq_len = input_ids.size(1)
        
        # If context is massive, chunk the prefill forward passes
        if seq_len > max_chunk_size:
            for i in range(0, seq_len - 1, max_chunk_size):
                end_idx = min(i + max_chunk_size, seq_len - 1)
                chunk = input_ids[:, i:end_idx]
                
                # Calculate position_ids for the chunk to maintain RoPE continuity
                position_ids = torch.arange(i, end_idx, dtype=torch.long, device=chunk.device).unsqueeze(0)
                
                with torch.no_grad():
                    self(input_ids=chunk, position_ids=position_ids, past_key_values=cache, use_cache=True)
            
            # Leave the remaining tokens to trigger the actual generation loop
            input_ids = input_ids[:, end_idx:]
            kwargs['position_ids'] = torch.arange(end_idx, seq_len, dtype=torch.long, device=input_ids.device).unsqueeze(0)
            
        kwargs['past_key_values'] = cache
        kwargs['use_cache'] = True
        return original_generate(input_ids=input_ids, **kwargs)

    model.generate = types.MethodType(custom_generate, model)
    
    # Prevent HF from overriding our StrataKVCache back to a basic DynamicCache
    original_prepare = model.prepare_inputs_for_generation
    def custom_prepare(self, input_ids, past_key_values=None, **kwargs):
        if past_key_values is None:
            past_key_values = StrataKVCache(config)
        return original_prepare(input_ids, past_key_values=past_key_values, **kwargs)
        
    model.prepare_inputs_for_generation = types.MethodType(custom_prepare, model)
    return model

# ==========================================
# 3. Save Utility (DDP Safe)
# ==========================================
def save_matrices_safely(accelerator, trainer_model, input_path, output_path, include_sonic=False):
    """Ensures only the main process writes to disk to prevent file corruption."""
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        matrices_dict = torch.load(input_path, map_location="cpu", weights_only=True) if os.path.exists(input_path) else {}
        unwrapped = accelerator.unwrap_model(trainer_model)
        for name, module in unwrapped.named_modules():
            # TransMLA Extractions
            if hasattr(module, "W_UK") and hasattr(module, "layer_idx"): 
                l_idx = module.layer_idx
                if l_idx not in matrices_dict: matrices_dict[l_idx] = {}
                matrices_dict[l_idx]["W_UK"] = module.W_UK.detach().cpu()
                matrices_dict[l_idx]["W_UV"] = module.W_UV.detach().cpu()
            if hasattr(module, "R_KV") and hasattr(module, "layer_idx"):
                matrices_dict[module.layer_idx]["R_KV"] = module.R_KV.detach().cpu()
                
            # SONIC Extractions
            if include_sonic and module.__class__.__name__ == "LlamaAttention" and hasattr(module, "sonic_cruncher"):
                l_idx = module.layer_idx
                if l_idx not in matrices_dict: matrices_dict[l_idx] = {}
                matrices_dict[l_idx]["sonic_cruncher_state"] = {k: v.detach().cpu() for k, v in module.sonic_cruncher.state_dict().items()}
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(matrices_dict, output_path)
        console.print(f"[bold blue]Saved weights successfully to {output_path}[/bold blue]")
    accelerator.wait_for_everyone()

# ==========================================
# 4. Master Pipeline
# ==========================================
def main():
    # Initialize Accelerator with Weights & Biases telemetry
    accelerator = Accelerator(log_with="wandb")
    
    MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct" # Change to target architecture base
    OUTPUT_DIR = "outputs/experiment_01"
    BASE_MATRICES = f"{OUTPUT_DIR}/base_transmla.pt"
    T2_MATRICES = f"{OUTPUT_DIR}/healed_t2.pt"
    T3_MATRICES = f"{OUTPUT_DIR}/healed_t3_sonic.pt"
    
    BATCH_SIZE = 1
    SEQ_LEN = 4096
    MAX_STEPS_T2 = 5000
    MAX_STEPS_T3 = 10000
    
    if accelerator.is_main_process:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        accelerator.init_trackers("StrataKV-E2E-Training", config={"model": MODEL_ID})
        console.print(f"[bold cyan]🚀 Starting StrataKV Deployment Pipeline on {accelerator.num_processes} GPUs[/bold cyan]")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # ---------------------------------------------------------
    # PHASE 1: Offline Extraction (Main Process Only)
    # ---------------------------------------------------------
    if accelerator.is_main_process and not os.path.exists(BASE_MATRICES):
        console.print("\n[bold magenta]=== Phase 1: TransMLA Calibration ===[/bold magenta]")
        run_offline_calibration(MODEL_ID, target_rank=128, rope_dim=64, num_samples=250, seq_len=1024, save_path=BASE_MATRICES)
    
    accelerator.wait_for_everyone() # BARRIER: Wait for GPU 0 to finish extracting
    
    # ---------------------------------------------------------
    # PHASE 2: Tier 2 Curriculum Healing (Multi-GPU)
    # ---------------------------------------------------------
    if not os.path.exists(T2_MATRICES):
        if accelerator.is_main_process: console.print("\n[bold magenta]=== Phase 2: Tier 2 TransMLA Healing ===[/bold magenta]")
        
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        t2_config = StrataKVConfig(
            tier0_size=4, tier1_size=1024, enable_tier0=True, enable_tier1=True, enable_tier2=True, enable_tier3=False,
            transmla_matrices_path=BASE_MATRICES,
            head_dim=base_model.config.hidden_size // base_model.config.num_attention_heads,
            num_kv_heads=getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads)
        )
        
        trainer_t2 = Tier2Trainer(base_model, t2_config)
        optimizer_t2 = torch.optim.AdamW(trainer_t2.get_trainable_parameters(), lr=2e-5)
        dataloader_t2 = build_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE)
        
        trainer_t2.model, optimizer_t2, dataloader_t2 = accelerator.prepare(trainer_t2.model, optimizer_t2, dataloader_t2)
        
        global_step = 0
        for batch_ids in dataloader_t2:
            if batch_ids is None or global_step >= MAX_STEPS_T2: break
            
            optimizer_t2.zero_grad()
            loss = trainer_t2.train_step(batch_ids.to(accelerator.device), prefix_len=2048)
            accelerator.backward(loss)
            optimizer_t2.step()
            
            if accelerator.is_main_process and global_step % 25 == 0:
                accelerator.log({"T2_Loss": loss.item()}, step=global_step)
                console.print(f"T2 Step {global_step}/{MAX_STEPS_T2} | Loss: {loss.item():.4f}")
            global_step += 1
            
        save_matrices_safely(accelerator, trainer_t2.model, BASE_MATRICES, T2_MATRICES, include_sonic=False)

        # Drop the trainer variables entirely to prevent DDP graph memory bleeds
        del trainer_t2, base_model, optimizer_t2, dataloader_t2
        torch.cuda.empty_cache()
        gc.collect()

    accelerator.wait_for_everyone()

    # ---------------------------------------------------------
    # PHASE 3: Tier 3 SONIC Distillation (Multi-GPU)
    # ---------------------------------------------------------
    if not os.path.exists(T3_MATRICES):
        if accelerator.is_main_process: console.print("\n[bold magenta]=== Phase 3: Tier 3 SONIC Distillation ===[/bold magenta]")
        
        base_model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
        t3_config = StrataKVConfig(
            tier0_size=4, tier1_size=256, tier2_size=1024, tier3_size=65536,
            enable_tier0=True, enable_tier1=True, enable_tier2=True, enable_tier3=True,
            transmla_matrices_path=T2_MATRICES,
            head_dim=base_model.config.hidden_size // base_model.config.num_attention_heads,
            num_kv_heads=getattr(base_model.config, "num_key_value_heads", base_model.config.num_attention_heads)
        )
        
        trainer_t3 = Tier3Trainer(base_model, t3_config, alpha_recon=1.0)
        optimizer_t3 = torch.optim.AdamW(trainer_t3.get_trainable_parameters(), lr=5e-5)
        dataloader_t3 = build_mixed_dataloader(tokenizer, SEQ_LEN, BATCH_SIZE)
        
        trainer_t3.model, optimizer_t3, dataloader_t3 = accelerator.prepare(trainer_t3.model, optimizer_t3, dataloader_t3)
        
        global_step = 0
        import random
        for batch_ids in dataloader_t3:
            if batch_ids is None or global_step >= MAX_STEPS_T3: break
            
            optimizer_t3.zero_grad()
            # Dynamic budgeting override (simulating fluctuating context clusters)
            loss, loss_dict = trainer_t3.train_step(
                batch_ids.to(accelerator.device), 
                prefix_len=2048, 
                k_budget=random.choice([2, 4]), 
                abit_threshold=random.uniform(0.3, 0.7)
            )
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(trainer_t3.get_trainable_parameters(), 1.0)
            optimizer_t3.step()
            
            if accelerator.is_main_process and global_step % 25 == 0:
                accelerator.log({"T3_Total": loss_dict["Total"], "T3_KD": loss_dict["L_KD"], "T3_Recon": loss_dict["L_Recon"]}, step=global_step)
                console.print(f"T3 Step {global_step}/{MAX_STEPS_T3} | KD: {loss_dict['L_KD']:.3f} | Recon: {loss_dict['L_Recon']:.3f}")
            global_step += 1
            
        save_matrices_safely(accelerator, trainer_t3.model, T2_MATRICES, T3_MATRICES, include_sonic=True)

    accelerator.wait_for_everyone()
    
    # ---------------------------------------------------------
    # PHASE 4: Benchmark Readiness Printout
    # ---------------------------------------------------------
    if accelerator.is_main_process:
        console.print("\n[bold green]🎉 Full Training Pipeline Completed! 🎉[/bold green]")
        console.print("\n[bold cyan]To run benchmarks (NeedleBench, RULER, MTBench101), inject StrataKV directly into their evaluation wrappers:[/bold cyan]")
        console.print("""
from transformers import AutoModelForCausalLM
from src.core.config import StrataKVConfig
from src.models.llama.modeling_llama import patch_llama_for_strata
from scripts.run_experiment import enable_benchmark_mode

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
config = StrataKVConfig(enable_tier2=True, enable_tier3=True, transmla_matrices_path="outputs/experiment_01/healed_t3_sonic.pt")

patch_llama_for_strata(model, config)
enable_benchmark_mode(model, config, max_chunk_size=2048)

# You can now feed `model` directly to `lm-eval-harness` or OpenCompass execution wrappers.
# The custom generation method will automatically chunk 128k context prefill to prevent OOMs!
        """)
    accelerator.end_training()

if __name__ == "__main__":
    main()
```

### How to execute this on your remote cluster

Because this script uses PyTorch's `accelerate` backend to automatically map Data Parallel models securely across GPUs, **do not run it using `python run_experiment.py`**. 

1. Ensure you are logged into wandb to track your distillation curves: `wandb login`
2. Run the configuration to setup your topology (Tell it how many GPUs you are using and set precision to `bf16`): `accelerate config`
3. Launch the unified pipeline: 
```bash
accelerate launch scripts/run_experiment.py
```

Once the run completes, simply paste the injection code snippet provided at the end of the console execution directly into the model-initialization logic of whatever benchmark library (like `lm-eval`) you are hooking into!