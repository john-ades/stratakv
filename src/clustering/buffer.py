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
        self.detector = StreamingABITBoundaryDetector(
            window_size=config.abit_window_size,
            min_split_tokens=config.abit_min_split_tokens,
            max_split_tokens=config.abit_max_split_tokens,
            initial_threshold=config.abit_initial_threshold
        )
        
        # Buffer the raw tensors until a seal event
        self._c_kv_buffer: List[torch.Tensor] = []
        self._k_rope_buffer: List[torch.Tensor] = []

    def _tensor_to_semantic_vector(self, c_kv_token: torch.Tensor) -> np.ndarray:
        """
        Extracts a 1D semantic vector representation from the c_kv token.
        Averages over num_heads if present.
        """
        if c_kv_token.dim() == 4:
            # (batch, num_heads, seq, dim)
            avg_token = c_kv_token.mean(dim=1)
        else:
            avg_token = c_kv_token
            
        # We flatten to 1D: [dim]. Assumes batch_size=1
        return avg_token.detach().cpu().to(torch.float32).numpy().flatten()
        

    def push(self, c_kv: torch.Tensor, k_rope: torch.Tensor) -> List[SealedCluster]:
        """
        Receives c_kv and k_rope tensors (usually chunked or single token).
        Runs them through the streaming ABIT detector.
        Returns a list of SealedCluster objects if boundaries are found.
        """
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
                
            # Convert to numpy semantic vector
            vec = self._tensor_to_semantic_vector(c_kv_token)
            
            # Step ABIT detector
            segment = self.detector.step(vec)
            
            # Remember to append tensors to the buffer!
            self._c_kv_buffer.append(c_kv_token)
            self._k_rope_buffer.append(k_rope_token)
            
            # If ABIT detector returns a segment before adding the current token:
            if segment is not None:
                # The segment returned DOES NOT include the current token yet.
                # So the tensors corresponding to the segment are everything EXCEPT the last buffered token.
                sealed = self._create_sealed_cluster(segment)
                sealed_clusters.append(sealed)
                
        return sealed_clusters

    def _create_sealed_cluster(self, segment: ClusterSegment) -> SealedCluster:
        """
        Packages the PyTorch tensor buffers corresponding to the returned numpy segment.
        """
        seq_dim_c = 2 if self._c_kv_buffer[0].dim() == 4 else 1
        seq_dim_k = 2 if self._k_rope_buffer[0].dim() == 4 else 1
        
        # We take exactly segment.total_tokens from the front of the list
        c_kv_segment = torch.cat(self._c_kv_buffer[:segment.total_tokens], dim=seq_dim_c)
        k_rope_segment = torch.cat(self._k_rope_buffer[:segment.total_tokens], dim=seq_dim_k)
        
        # Remove from buffer
        self._c_kv_buffer = self._c_kv_buffer[segment.total_tokens:]
        self._k_rope_buffer = self._k_rope_buffer[segment.total_tokens:]

        return SealedCluster(
            c_kv=c_kv_segment,
            k_rope=k_rope_segment,
            segment=segment
        )

    def flush(self) -> Optional[SealedCluster]:
        """
        Force flush the underlying ABIT detector.
        """
        segment = self.detector.flush()
        if segment is not None:
            return self._create_sealed_cluster(segment)
        return None
