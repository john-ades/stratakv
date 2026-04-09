import os
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import LlamaForCausalLM, AutoTokenizer
from datasets import load_dataset
from typing import Tuple, Dict, Any

def calculate_rorope(K_x: torch.Tensor, K_y: torch.Tensor) -> torch.Tensor:
    """
    Computes orthogonal rotation matrix U_l for a RoPE subspace.
    K_x, K_y: [num_tokens, current_dim]
    """
    # 1. Store original dtype (float16/bfloat16) to cast back later
    orig_dtype = K_x.dtype
    
    # 2. UPCAST to float32 BEFORE matrix multiplication to prevent massive overflow
    K_x_f32 = K_x.float()
    K_y_f32 = K_y.float()
    
    # Expected dimensions: [N, D] where D is the dimension of the subspace
    sigma_x = K_x_f32.T @ K_x_f32
    sigma_y = K_y_f32.T @ K_y_f32
    sigma_sum = sigma_x + sigma_y
    
    # Eigendecomposition to maximize variance 
    eigenvalues, eigenvectors = torch.linalg.eigh(sigma_sum)
    
    # Sort descending
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    U_l = eigenvectors[:, sorted_indices]
    
    # 3. Cast back to original dtype for downstream LLaMA compatibility
    return U_l.to(orig_dtype)

def calculate_bkv_pca(K_nope: torch.Tensor, V: torch.Tensor, target_rank: int) -> Tuple[torch.Tensor, float]:
    """
    Performs Balanced KV Joint PCA.
    K_nope: [num_tokens, nope_dim]
    V: [num_tokens, v_dim]
    Returns PCA projection matrix R_KV and scaling factor alpha.
    """
    K_nope = K_nope.float()
    V = V.float()
    
    # E_t [ || K_nope ||_2^2 ]
    k_nope_norm_sq = torch.mean(torch.sum(K_nope ** 2, dim=-1))
    v_norm_sq = torch.mean(torch.sum(V ** 2, dim=-1))
    
    # Calculate balance factor alpha
    alpha = torch.sqrt(k_nope_norm_sq / (v_norm_sq + 1e-8)).item()
    
    K_nope_scaled = K_nope / alpha
    
    # Concatenate BKV NoPE and Values
    C_nope = torch.cat([K_nope_scaled, V], dim=-1)
    
    # PCA on C_nope
    C_nope_centered = C_nope - C_nope.mean(dim=0, keepdim=True)
    
    # Covariance matrix (scale by 1 / (N - 1))
    cov = C_nope_centered.T @ C_nope_centered / (C_nope_centered.size(0) - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(cov)
    
    # Sort descending
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    eigenvectors = eigenvectors[:, sorted_indices]
    
    R_KV = eigenvectors[:, :target_rank]
    
    return R_KV, alpha


def harvest_activations(model: nn.Module, tokenizer: AutoTokenizer, dataset, num_samples: int, seq_len: int) -> Tuple[Dict, Dict]:
    """
    Runs forwards passes to harvest un-RoPE'd keys and values from the base LLaMA model.
    """
    device = next(model.parameters()).device
    model.eval()
    
    all_K = {layer_idx: [] for layer_idx in range(model.config.num_hidden_layers)}
    all_V = {layer_idx: [] for layer_idx in range(model.config.num_hidden_layers)}
    
    hooks = []
    
    # Hook k_proj and v_proj directly inside LlamaAttention
    for layer_idx, layer in enumerate(model.model.layers):
        def k_hook(module, inputs, output, l_idx=layer_idx):
            all_K[l_idx].append(output.detach().cpu())
        def v_hook(module, inputs, output, l_idx=layer_idx):
            all_V[l_idx].append(output.detach().cpu())
            
        hooks.append(layer.self_attn.k_proj.register_forward_hook(k_hook))
        hooks.append(layer.self_attn.v_proj.register_forward_hook(v_hook))

    # Prepare dataset
    texts = [t for t in dataset["text"] if len(t.strip()) > 0]
    
    with torch.no_grad():
        tokenized_so_far = []
        text_idx = 0
        
        for i in tqdm(range(num_samples), desc="Harvesting Activations"):
            # Ensure we have enough tokens to form a sequence
            while len(tokenized_so_far) < seq_len and text_idx < len(texts):
                tokens = tokenizer(texts[text_idx], add_special_tokens=False).input_ids
                tokenized_so_far.extend(tokens)
                text_idx += 1
                
            if len(tokenized_so_far) < seq_len:
                print("Reached end of dataset prematurely.")
                break
                
            # Extract the correct chunk length
            chunk = tokenized_so_far[:seq_len]
            tokenized_so_far = tokenized_so_far[seq_len:]
            
            input_ids = torch.tensor([chunk], dtype=torch.long, device=device)
            # Forward pass: we only need the hooked states, ignore the final outputs
            model(input_ids)
            
    for hook in hooks:
        hook.remove()
        
    for l_idx in range(model.config.num_hidden_layers):
        all_K[l_idx] = torch.cat(all_K[l_idx], dim=0)
        all_V[l_idx] = torch.cat(all_V[l_idx], dim=0)
        
    return all_K, all_V


def extract_transmla_matrices_for_layer(
    layer_idx: int,
    K_layer: torch.Tensor, 
    V_layer: torch.Tensor, 
    num_kv_heads: int, 
    head_dim: int,
    target_rank: int,
    rope_retained_dim: int = 64
) -> Dict[str, Any]:
    """
    Processes all offline extracted matrices for a single layer.
    """
    if torch.cuda.is_available():
        K_layer, V_layer = K_layer.cuda(), V_layer.cuda()

    # K_layer represents the raw activations from k_proj: [num_batches, seq_len, num_kv_heads * head_dim]
    batch_size, seq_len, _ = K_layer.shape
    total_tokens = batch_size * seq_len
    
    # Reshape to [total_tokens, num_kv_heads, head_dim]
    K_reshaped = K_layer.view(total_tokens, num_kv_heads, head_dim)
    V_reshaped = V_layer.view(total_tokens, num_kv_heads, head_dim)
    
    # HF RoPE splits the head_dim into two halves logic (x1, x2)
    half_dim = head_dim // 2
    K_x = K_reshaped[..., :half_dim]
    K_y = K_reshaped[..., half_dim:]
    
    # For simplicity across heads, we flatten the token and head dimensions or do it per head
    # TransMLA calculates U_l per subspace. Here we do it globally across heads for the NoPE alignment.
    K_x_flat = K_x.reshape(-1, half_dim)
    K_y_flat = K_y.reshape(-1, half_dim)
    
    U_l = calculate_rorope(K_x_flat, K_y_flat)
    
    # Apply rotation
    K_x_rotated = K_x @ U_l.T
    K_y_rotated = K_y @ U_l.T
    K_rotated = torch.cat([K_x_rotated, K_y_rotated], dim=-1)
    
    # Separate RoPE and NoPE
    K_rope = K_rotated[..., :rope_retained_dim]
    K_nope = K_rotated[..., rope_retained_dim:]
    
    # Flatten NOPE and V across heads to perform Joint BKV PCA
    K_nope_flat = K_nope.reshape(total_tokens, -1)
    V_flat = V_reshaped.reshape(total_tokens, -1)
    
    R_KV, alpha = calculate_bkv_pca(K_nope_flat, V_flat, target_rank)
    
    return {
        "layer_idx": layer_idx,
        "U_l": U_l.cpu(),
        "alpha": alpha,
        "R_KV": R_KV.cpu(),
        "K_rope_dim": rope_retained_dim,
        "target_rank": target_rank
    }


def run_offline_calibration(
    model_id: str,
    target_rank: int = 128,
    rope_dim: int = 64,
    num_samples: int = 50,
    seq_len: int = 1024,
    save_path: str = "outputs/transmla_matrices.pt"
):
    """
    Orchestrates the offline calibration process for LLaMA TransMLA extraction.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Loading tokenizer and model: {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Prevent device_map="auto" from causing NCCL deadlocks in Accelerate DDP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_rank = os.environ.get("LOCAL_RANK", None)
    
    if local_rank is not None:
        device_map = {"": f"cuda:{local_rank}"}
    else:
        device_map = {"": device} if torch.cuda.is_available() else "auto"
        
    model = LlamaForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch.float16)
    
    print("Loading calibration dataset (WikiText-2)...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    
    print("Harvesting activations (Forward Passes)...")
    all_K, all_V = harvest_activations(model, tokenizer, dataset, num_samples, seq_len)
    
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.hidden_size // model.config.num_attention_heads
    
    extracted_data = {}
    
    print("Computing RoRoPE and BKV PCA specifically per layer...")
    for layer_idx in tqdm(range(model.config.num_hidden_layers), desc="Extracting Matrices"):
        layer_K = all_K[layer_idx]
        layer_V = all_V[layer_idx]
        
        matrices = extract_transmla_matrices_for_layer(
            layer_idx=layer_idx,
            K_layer=layer_K,
            V_layer=layer_V,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            target_rank=target_rank,
            rope_retained_dim=rope_dim
        )
        extracted_data[layer_idx] = matrices
        
    print(f"Saving extracted TransMLA matrices to {save_path}")
    torch.save(extracted_data, save_path)
    print("Phase 1 Offline Calibration Complete.")
    
if __name__ == "__main__":
    # Example usage for testing locally
    run_offline_calibration(
        model_id="mlx-community/Llama-3.2-1B-Instruct", # Example model ID matching user habits
        target_rank=128,
        num_samples=10, # Short sample for smoke testing
        seq_len=512,
        save_path="outputs/llama_transmla_v1.pt"
    )
