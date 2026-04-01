import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SonicCruncher(nn.Module):
    """
    Tier 3 Orthogonal Sequence Compression (SONIC).
    Compresses an N-length sequence of TransMLA latents (C_kv) into K Nexus base tokens
    using an attention-weighted information bottleneck.
    """
    def __init__(self, dim: int, max_k: int = 4):
        super().__init__()
        self.dim = dim
        self.max_k = max_k
        
        # Learnable spatial anchors for the Nexus tokens
        self.nexus_base = nn.Parameter(torch.randn(max_k, dim))
        
        # Lightweight projection matrices for the Information Bottleneck
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def forward(self, c_kv: torch.Tensor, k: int) -> torch.Tensor:
        """
        Compresses `c_kv` into `k` nexus tokens.
        
        Args:
            c_kv: The sealed N-length TransMLA latents. 
                  Expected shape: (batch, num_heads, seq_len, dim) or (batch, seq_len, dim)
            k: The dynamic budget of Nexus tokens to allocate for this cluster (1 <= k <= max_k).
            
        Returns:
            c_nexus: The compressed latent representation.
                     Shape matches spatial rank of input: (batch, num_heads, k, dim) or (batch, k, dim)
        """
        print(f"SONIC CALLED: seq_len={c_kv.shape[-2]} training={self.training}")
        assert 1 <= k <= self.max_k, f"k must be between 1 and {self.max_k}, got {k}"
        
        is_3d = False
        if c_kv.dim() == 3:
            c_kv = c_kv.unsqueeze(1)  # (batch, 1, seq_len, dim)
            is_3d = True
            
        batch_size, num_heads, seq_len, dim = c_kv.shape
        assert dim == self.dim, f"Expected hidden dimension {self.dim}, got {dim}"
        
        # 1. Fetch the active $K$ subset of nexus base tokens
        active_nexus = self.nexus_base[:k, :] # (k, dim)
        
        # 2. Expand to match batch and num_heads: (batch, num_heads, k, dim)
        active_nexus = active_nexus.view(1, 1, k, dim).expand(batch_size, num_heads, k, dim)
        
        # 3. Projection to attention space
        q = self.q_proj(active_nexus) # (batch, num_heads, k, dim)
        k_states = self.k_proj(c_kv)  # (batch, num_heads, seq_len, dim)
        v_states = self.v_proj(c_kv)  # (batch, num_heads, seq_len, dim)
        
        # 4. Attention-weighted aggregation (Information Bottleneck)
        # q: (..., k, dim), k_states: (..., seq_len, dim) -> attn: (..., k, seq_len)
        attn_weights = torch.matmul(q, k_states.transpose(-1, -2)) / math.sqrt(dim)
        attn_probs = F.softmax(attn_weights, dim=-1)
        
        # attn_probs: (..., k, seq_len), v_states: (..., seq_len, dim) -> c_nexus: (..., k, dim)
        c_nexus = torch.matmul(attn_probs, v_states)
        
        # 5. Output projection
        c_nexus_out = self.o_proj(c_nexus)
        
        # 6. Reconstruction Loss (Phase 5 Healing)
        if self.training:
            # Reconstruct original kv from nexus using the inverse of the bottleneck probabilities
            c_kv_recon = torch.matmul(attn_probs.transpose(-1, -2), c_nexus)
            loss = F.mse_loss(c_kv_recon, c_kv.detach())
            if not hasattr(self, 'recon_loss') or self.recon_loss is None:
                self.recon_loss = loss
            else:
                self.recon_loss = self.recon_loss + loss
        
        if is_3d:
            c_nexus_out = c_nexus_out.squeeze(1) # Revert to (batch, k, dim)
            
        return c_nexus_out
