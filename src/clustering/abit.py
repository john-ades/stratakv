import numpy as np
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class ClusterSegment:
    """
    A dynamically sealed segment of continuous semantic memory.
    Ready to be consumed by the Tier 3 SONIC Cruncher.
    """
    embeddings: np.ndarray
    token_counts: List[int]
    total_tokens: int
    medoid_idx: int  # The local sequence index of the Saliency Anchor
    
    @property
    def medoid_embedding(self) -> np.ndarray:
        """Returns the specific latent vector acting as the Saliency Anchor."""
        return self.embeddings[self.medoid_idx]


class StreamingABITBoundaryDetector:
    """
    Adaptive Binary Iterative Threshold (ABIT) Clustering - Online Streaming Version.
    
    Stripped of hierarchical recursion, focused purely on sequential semantic 
    boundary detection for infinite context streams. Includes native Medoid
    saliency anchoring for SONIC's positional binding.
    """
    def __init__(
        self,
        window_size: int = 3,
        min_split_tokens: int = 5,
        max_split_tokens: int = 128,
        split_tokens_tolerance: int = 5,
        threshold_adjustment: float = 0.01,
        initial_threshold: float = 0.5,
    ):
        self.window_size = window_size
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.threshold_adjustment = threshold_adjustment
        
        # Dynamic threshold that adapts continuously
        self.current_threshold = initial_threshold

        # Buffer for the active (forming) cluster segment
        self.active_embeddings: List[np.ndarray] = []
        self.active_counts: List[int] = []
        self.active_total_tokens: int = 0

    def process_batch(self, X: np.ndarray, T: Optional[np.ndarray] = None) -> List[ClusterSegment]:
        """
        Process a batch of embeddings (e.g., during LLM Prefill).
        Returns a list of sealed cluster segments found within the batch.
        """
        segments = []
        if T is None:
            T = np.ones(X.shape[0], dtype=int)
            
        for i in range(X.shape[0]):
            seg = self.step(X[i], T[i])
            if seg is not None:
                segments.append(seg)
        return segments

    def step(self, embedding: np.ndarray, count: int = 1) -> Optional[ClusterSegment]:
        """
        Ingest a single token/chunk embedding incrementally (e.g., during Autoregressive Decoding).
        Returns a sealed ClusterSegment if a semantic boundary is triggered *before* this token.
        """
        # If buffer is empty, initialize the new cluster segment
        if not self.active_embeddings:
            self._append_to_active(embedding, count)
            return None

        # Calculate rolling similarity against the immediate past context within this cluster
        window_start = max(0, len(self.active_embeddings) - self.window_size)
        recent_embs = self.active_embeddings[window_start:]
        cumulative_context = np.mean(recent_embs, axis=0)
        
        norm_ctx = np.linalg.norm(cumulative_context)
        norm_emb = np.linalg.norm(embedding)
        
        if norm_ctx < 1e-10 or norm_emb < 1e-10:
            sim = 0.0
        else:
            sim = np.dot(cumulative_context, embedding) / (norm_ctx * norm_emb)

        # Boundary Evaluation
        force_split = (self.active_total_tokens + count > self.max_split_tokens)
        semantic_split = (self.active_total_tokens >= self.min_split_tokens and sim < self.current_threshold)

        if force_split or semantic_split:
            # 1. Seal the current cluster BEFORE appending the new token
            sealed_cluster = self._seal_cluster()

            # 2. Adaptive Thresholding (Temporal alternative to original ABIT's binary search)
            if force_split:
                # Hitting the max token limit implies threshold is too permissive (too low).
                # Increase it so it triggers a natural semantic split earlier next time.
                self.current_threshold = min(1.0, self.current_threshold + self.threshold_adjustment)
            elif semantic_split:
                # If we split semantically but very close to the minimum limit, 
                # threshold might be too aggressive (too high). Relax it.
                if sealed_cluster.total_tokens <= self.min_split_tokens + self.split_tokens_tolerance:
                    self.current_threshold = max(-1.0, self.current_threshold - self.threshold_adjustment)

            # 3. Initialize the next cluster segment with the incoming token
            self._append_to_active(embedding, count)
            return sealed_cluster
        else:
            self._append_to_active(embedding, count)
            return None

    def _append_to_active(self, embedding: np.ndarray, count: int):
        self.active_embeddings.append(embedding)
        self.active_counts.append(count)
        self.active_total_tokens += count

    def _seal_cluster(self) -> ClusterSegment:
        """
        Seals the active embeddings into a cluster and computes the Saliency Anchor (Medoid).
        """
        embeddings_arr = np.array(self.active_embeddings)
        counts_list = list(self.active_counts)

        # -------------------------------------------------------------
        # Saliency Anchoring / Medoid Extraction
        # Finds the specific token acting as the semantic "center of mass",
        # providing absolute positional integrity for the SONIC Nexus tokens.
        # -------------------------------------------------------------
        n_tokens = embeddings_arr.shape[0]
        if n_tokens > 1:
            # Normalize for fast Cosine Similarity via Dot Product
            norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized_embeddings = embeddings_arr / norms
            
            # Pairwise similarity matrix
            sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            # The Medoid is the token with the highest average similarity to all peers
            avg_sims = np.mean(sim_matrix, axis=1)
            medoid_idx = int(np.argmax(avg_sims))
        else:
            medoid_idx = 0

        segment = ClusterSegment(
            embeddings=embeddings_arr,
            token_counts=counts_list,
            total_tokens=self.active_total_tokens,
            medoid_idx=medoid_idx
        )

        # Reset active cluster buffer
        self.active_embeddings.clear()
        self.active_counts.clear()
        self.active_total_tokens = 0

        return segment

    def flush(self) -> Optional[ClusterSegment]:
        """
        Forces the remaining buffer to seal and emit. 
        Useful at the end of a generation stream or document processing.
        """
        if self.active_embeddings:
            return self._seal_cluster()
        return None