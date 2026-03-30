import torch
import torch.nn as nn
from typing import Tuple, Optional

class TransMLACruncher(nn.Module):
    """
    Handles the compression of dense GQA KV tokens (Tier 1) into 
    latent TransMLA tokens (Tier 2).
    """
    def __init__(
        self, 
        layer_idx: int,
        num_kv_heads: int,
        head_dim: int,
        rope_retained_dim: int,
        target_rank: int,
        matrices_path: Optional[str] = None
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.half_dim = head_dim // 2
        
        self.rope_retained_dim = rope_retained_dim
        self.target_rank = target_rank
        
        # Extracted Matrices
        # U_l: Rotation matrix for RoPE [half_dim, half_dim]
        self.register_buffer("U_l", torch.eye(self.half_dim))
        
        # R_KV: Down-projection matrix from PCA 
        nope_dim = head_dim - rope_retained_dim
        v_dim = head_dim
        
        # PCA was trained on concatenated and flattened NOPE and V across heads
        in_features = nope_dim * num_kv_heads + v_dim * num_kv_heads
        
        self.register_buffer("R_KV", torch.zeros(in_features, target_rank))
        self.register_buffer("alpha", torch.ones(1))
        
        if matrices_path is not None:
            self.load_matrices(matrices_path)
            
    def load_matrices(self, path: str):
        data = torch.load(path, map_location="cpu", weights_only=False)
        if self.layer_idx in data:
            layer_data = data[self.layer_idx]
            self.U_l.copy_(layer_data["U_l"])
            self.R_KV.copy_(layer_data["R_KV"])
            
            # Ensure alpha is a 1D tensor
            alpha_val = layer_data["alpha"]
            if isinstance(alpha_val, float):
                self.alpha.copy_(torch.tensor([alpha_val]))
            else:
                self.alpha.copy_(torch.tensor(alpha_val).view(1))
            
    def forward(self, K_evicted: torch.Tensor, V_evicted: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compresses dropped tokens from Tier 1.
        K_evicted: [batch, num_kv_heads, seq_len, head_dim]
        V_evicted: [batch, num_kv_heads, seq_len, head_dim]
        
        Returns:
            C_kv: [batch, seq_len, target_rank]
            K_rope: [batch, num_kv_heads, seq_len, rope_retained_dim]
        """
        batch_size, num_kv_heads, seq_len, head_dim = K_evicted.shape
        
        # Permute to sequence-major for easier flattening later: [batch, seq_len, num_kv_heads, head_dim]
        K_perm = K_evicted.permute(0, 2, 1, 3) 
        
        # HF RoPE splits the head_dim into two halves logic (x1, x2)
        K_x = K_perm[..., :self.half_dim]
        K_y = K_perm[..., self.half_dim:]
        
        # Apply RoRoPE rotation U_l
        K_x_rotated = K_x @ self.U_l.T
        K_y_rotated = K_y @ self.U_l.T
        
        K_rotated = torch.cat([K_x_rotated, K_y_rotated], dim=-1)
        
        # Separate RoPE and NoPE
        K_rope = K_rotated[..., :self.rope_retained_dim] # [batch, seq_len, num_kv_heads, rope_retained_dim]
        K_nope = K_rotated[..., self.rope_retained_dim:] # [batch, seq_len, num_kv_heads, nope_dim]
        
        # BKV-PCA Compression
        # Scale K_nope by alpha balancing factor
        K_nope_scaled = K_nope / self.alpha
        
        # Flatten across heads for joint compression
        # K_nope_scaled: [batch, seq_len, num_kv_heads * nope_dim]
        K_nope_flat = K_nope_scaled.reshape(batch_size, seq_len, -1)
        
        # V_evicted: [batch, num_kv_heads, seq_len, head_dim] -> [batch, seq_len, num_kv_heads * head_dim]
        V_perm = V_evicted.permute(0, 2, 1, 3)
        V_flat = V_perm.reshape(batch_size, seq_len, -1)
        
        # Concatenate NoPE and Values
        C_nope = torch.cat([K_nope_flat, V_flat], dim=-1) # [batch, seq_len, in_features]
        
        # Matrix multiply with PCA down-projection
        C_kv = C_nope @ self.R_KV # [batch, seq_len, target_rank]
        
        # Re-permute K_rope back to expected structural cache format [batch, num_kv_heads, seq_len, rope_retained_dim]
        K_rope = K_rope.permute(0, 2, 1, 3)
        
        return C_kv, K_rope
