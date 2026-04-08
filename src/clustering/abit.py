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
        batch_size: int = 1,
        window_size: int = 3,
        min_split_tokens: int = 5,
        max_split_tokens: int = 128,
        split_tokens_tolerance: int = 5,
        threshold_adjustment: float = 0.01,
        initial_threshold: float = 0.5,
    ):
        self.batch_size = batch_size
        self.window_size = window_size
        self.min_split_tokens = min_split_tokens
        self.max_split_tokens = max_split_tokens
        self.split_tokens_tolerance = split_tokens_tolerance
        self.threshold_adjustment = threshold_adjustment
        
        # Dynamic threshold that adapts continuously per batch sequence
        self.current_threshold = [initial_threshold for _ in range(batch_size)]

        # Buffer for the active (forming) cluster segments per batch sequence
        self.active_embeddings: List[List[np.ndarray]] = [[] for _ in range(batch_size)]
        self.active_counts: List[List[int]] = [[] for _ in range(batch_size)]
        self.active_total_tokens: List[int] = [0 for _ in range(batch_size)]

    def process_batch(self, X: np.ndarray, T: Optional[np.ndarray] = None) -> List[List[ClusterSegment]]:
        """
        Process a batch of embeddings (e.g., during LLM Prefill).
        X has shape (batch_size, seq_len, dim) or (seq_len, dim).
        Returns a list of length batch_size, where each element is a list of sealed cluster segments found.
        """
        if X.ndim == 2:
            # (seq_len, dim) -> (1, seq_len, dim)
            X = np.expand_dims(X, axis=0)
            
        assert X.shape[0] == self.batch_size, f"Input batch size {X.shape[0]} does not match initialized batch_size {self.batch_size}"
        
        segments = [[] for _ in range(self.batch_size)]
        
        seq_len = X.shape[1]
        for i in range(seq_len):
            if T is None:
                step_counts = 1
            elif T.ndim == 1:
                step_counts = T[i]
            else:
                step_counts = T[:, i]
            
            step_segments = self.step(X[:, i, :], step_counts)
            for b in range(self.batch_size):
                if step_segments[b] is not None:
                    segments[b].append(step_segments[b])
                    
        return segments

    def step(self, embeddings: np.ndarray, count: int = 1) -> List[Optional[ClusterSegment]]:
        """
        Ingest a single token/chunk embeddings incrementally for the batch.
        embeddings shape: (batch_size, dim) or (dim) if batch_size=1
        Returns a list of length batch_size containing sealed ClusterSegment or None.
        """
        if embeddings.ndim == 1:
            embeddings = np.expand_dims(embeddings, axis=0)
            
        assert embeddings.shape[0] == self.batch_size, f"Input batch size {embeddings.shape[0]} != detector batch_size {self.batch_size}"
        
        results = []
        for b in range(self.batch_size):
            emb = embeddings[b]
            
            # If buffer is empty, initialize the new cluster segment
            if not self.active_embeddings[b]:
                self._append_to_active(b, emb, count)
                results.append(None)
                continue

            # Calculate rolling similarity against the immediate past context within this cluster
            window_start = max(0, len(self.active_embeddings[b]) - self.window_size)
            recent_embs = self.active_embeddings[b][window_start:]
            cumulative_context = np.mean(recent_embs, axis=0)
            
            norm_ctx = np.linalg.norm(cumulative_context)
            norm_emb = np.linalg.norm(emb)
            
            if norm_ctx < 1e-10 or norm_emb < 1e-10:
                sim = 0.0
            else:
                sim = np.dot(cumulative_context, emb) / (norm_ctx * norm_emb)

            # Boundary Evaluation
            force_split = (self.active_total_tokens[b] + count > self.max_split_tokens)
            semantic_split = (self.active_total_tokens[b] >= self.min_split_tokens and sim < self.current_threshold[b])

            if force_split or semantic_split:
                # 1. Seal the current cluster BEFORE appending the new token
                sealed_cluster = self._seal_cluster(b)

                # 2. Adaptive Thresholding
                if force_split:
                    self.current_threshold[b] = min(1.0, self.current_threshold[b] + self.threshold_adjustment)
                elif semantic_split:
                    if sealed_cluster.total_tokens <= self.min_split_tokens + self.split_tokens_tolerance:
                        self.current_threshold[b] = max(-1.0, self.current_threshold[b] - self.threshold_adjustment)

                # 3. Initialize the next cluster segment with the incoming token
                self._append_to_active(b, emb, count)
                results.append(sealed_cluster)
            else:
                self._append_to_active(b, emb, count)
                results.append(None)
                
        return results

    def _append_to_active(self, b: int, embedding: np.ndarray, count: int):
        self.active_embeddings[b].append(embedding)
        self.active_counts[b].append(count)
        self.active_total_tokens[b] += count

    def _seal_cluster(self, b: int) -> ClusterSegment:
        """
        Seals the active embeddings into a cluster and computes the Saliency Anchor (Medoid) for batch item b.
        """
        embeddings_arr = np.array(self.active_embeddings[b])
        counts_list = list(self.active_counts[b])

        n_tokens = embeddings_arr.shape[0]
        if n_tokens > 1:
            norms = np.linalg.norm(embeddings_arr, axis=1, keepdims=True)
            norms[norms == 0] = 1e-10
            normalized_embeddings = embeddings_arr / norms
            
            sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
            
            avg_sims = np.mean(sim_matrix, axis=1)
            medoid_idx = int(np.argmax(avg_sims))
        else:
            medoid_idx = 0

        segment = ClusterSegment(
            embeddings=embeddings_arr,
            token_counts=counts_list,
            total_tokens=self.active_total_tokens[b],
            medoid_idx=medoid_idx
        )

        # Reset active cluster buffer
        self.active_embeddings[b].clear()
        self.active_counts[b].clear()
        self.active_total_tokens[b] = 0

        return segment

    def flush(self) -> List[Optional[ClusterSegment]]:
        """
        Forces the remaining buffer to seal and emit for all batch sequences.
        Useful at the end of a generation stream or document processing.
        """
        results = []
        for b in range(self.batch_size):
            if self.active_embeddings[b]:
                results.append(self._seal_cluster(b))
            else:
                results.append(None)
        return results