import torch
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .abit import StreamingABITBoundaryDetector, ClusterSegment

@dataclass
class SealedCluster:
    """
    A unified sealed cluster that holds both the underlying semantic
    boundary information (ClusterSegment) and the corresponding PyTorch tensors.
    """
    c_kv: torch.Tensor
    k_rope: torch.Tensor
    segment: ClusterSegment
    batch_idx: int = 0
    
    @property
    def medoid_idx(self) -> int:
        return self.segment.medoid_idx

    @property
    def medoid_k_rope(self) -> torch.Tensor:
        """
        Phase 2: Positional Harvesting.
        Extracts the decoupled positional embedding tensor (K_rope) of the specific Medoid token.
        Returns a tensor of shape matching a single token but retaining the sequence dimension.
        Works with both 3D (batch, seq, dim) and 4D (batch, heads, seq, dim) formats.
        """
        seq_dim = 2 if self.k_rope.dim() == 4 else 1
        return self.k_rope.select(seq_dim, self.medoid_idx).unsqueeze(seq_dim)

    def expand_medoid_k_rope(self, num_nexus_tokens: int) -> torch.Tensor:
        """
        Phase 2: RoPE Injection.
        Explicitly maps the Medoid's K_rope to the K Nexus tokens by expanding it
        along the sequence dimension.
        """
        medoid_rope = self.medoid_k_rope
        seq_dim = 2 if self.k_rope.dim() == 4 else 1
        
        # Expand along the sequence dimension
        expand_shape = list(medoid_rope.shape)
        expand_shape[seq_dim] = num_nexus_tokens
        
        return medoid_rope.expand(*expand_shape).contiguous()


class AbitClusterBuffer:
    """
    Acts as a bridge between the PyTorch Tier 2 Latent eviction and the 
    numpy-based StreamingABITBoundaryDetector.
    """
    def __init__(self, config):
        self.config = config
        self.detector = None
        
        self._c_kv_buffer: List[List[torch.Tensor]] = []
        self._k_rope_buffer: List[List[torch.Tensor]] = []

    def _ensure_initialized(self, batch_size: int):
        if self.detector is None or getattr(self.detector, 'batch_size', None) != batch_size:
            self.detector = StreamingABITBoundaryDetector(
                batch_size=batch_size,
                window_size=self.config.abit_window_size,
                min_split_tokens=self.config.abit_min_split_tokens,
                max_split_tokens=self.config.abit_max_split_tokens,
                initial_threshold=self.config.abit_initial_threshold
            )
            self._c_kv_buffer = [[] for _ in range(batch_size)]
            self._k_rope_buffer = [[] for _ in range(batch_size)]

    def _tensor_to_semantic_vector(self, c_kv_token: torch.Tensor) -> np.ndarray:
        """
        Extracts a 1D or 2D semantic vector representation from the c_kv token.
        Averages over num_heads if present.
        """
        if c_kv_token.dim() == 4:
            # (batch, num_heads, seq, dim)
            avg_token = c_kv_token.mean(dim=1)
        else:
            avg_token = c_kv_token
            
        # Squeeze seq dim if 1: (batch, 1, dim) -> (batch, dim)
        if avg_token.dim() == 3 and avg_token.shape[1] == 1:
            avg_token = avg_token.squeeze(1)
            
        return avg_token.detach().cpu().to(torch.float32).numpy()

    def push(self, c_kv: torch.Tensor, k_rope: torch.Tensor) -> List[SealedCluster]:
        """
        Receives c_kv and k_rope tensors (usually chunked or single token).
        Runs them through the streaming ABIT detector.
        Returns a list of SealedCluster objects if boundaries are found.
        """
        batch_size = c_kv.shape[0]
        self._ensure_initialized(batch_size)
        
        sealed_clusters = []
        seq_dim_c = 2 if c_kv.dim() == 4 else 1
        seq_dim_k = 2 if k_rope.dim() == 4 else 1

        seq_len = c_kv.shape[seq_dim_c]
        
        for i in range(seq_len):
            # Extract i-th token along sequence dimension
            if seq_dim_c == 2:
                c_kv_token = c_kv[:, :, i:i+1, :]
            else:
                c_kv_token = c_kv[:, i:i+1, :]
                
            if seq_dim_k == 2:
                k_rope_token = k_rope[:, :, i:i+1, :]
            else:
                k_rope_token = k_rope[:, i:i+1, :]
                
            # Convert to numpy semantic vector (batch_size, dim)
            vec = self._tensor_to_semantic_vector(c_kv_token)
            
            # Step ABIT detector
            segments = self.detector.step(vec)
            
            for b in range(batch_size):
                # Append tensors to the buffer per batch item
                self._c_kv_buffer[b].append(c_kv_token[b:b+1])
                self._k_rope_buffer[b].append(k_rope_token[b:b+1])
                
                # If ABIT detector returns a segment before adding the current token:
                if segments[b] is not None:
                    sealed = self._create_sealed_cluster(b, segments[b])
                    sealed_clusters.append(sealed)
                
        return sealed_clusters

    def _create_sealed_cluster(self, b: int, segment: ClusterSegment) -> SealedCluster:
        """
        Packages the PyTorch tensor buffers corresponding to the returned numpy segment.
        """
        seq_dim_c = 2 if self._c_kv_buffer[b][0].dim() == 4 else 1
        seq_dim_k = 2 if self._k_rope_buffer[b][0].dim() == 4 else 1
        
        # We take exactly segment.total_tokens from the front of the list
        c_kv_segment = torch.cat(self._c_kv_buffer[b][:segment.total_tokens], dim=seq_dim_c)
        k_rope_segment = torch.cat(self._k_rope_buffer[b][:segment.total_tokens], dim=seq_dim_k)
        
        # Remove from buffer
        self._c_kv_buffer[b] = self._c_kv_buffer[b][segment.total_tokens:]
        self._k_rope_buffer[b] = self._k_rope_buffer[b][segment.total_tokens:]

        return SealedCluster(
            c_kv=c_kv_segment,
            k_rope=k_rope_segment,
            segment=segment,
            batch_idx=b
        )

    def flush(self) -> List[SealedCluster]:
        """
        Force flush the underlying ABIT detector.
        """
        sealed_clusters = []
        if self.detector is not None:
            segments = self.detector.flush()
            for b, seg in enumerate(segments):
                if seg is not None:
                    sealed_clusters.append(self._create_sealed_cluster(b, seg))
        return sealed_clusters
