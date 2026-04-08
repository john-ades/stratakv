import pytest
import torch
import numpy as np
from src.core.config import StrataKVConfig
from src.clustering.abit import StreamingABITBoundaryDetector
from src.clustering.buffer import ClusterSegment

def test_batch_packing_isolation():
    """
    Verifies that ABIT Boundary Detection safely processes a batch dimension where
    sequence A triggers an acoustic boundary but sequence B does not.
    """
    batch_size = 2
    head_dim = 16
    
    detector = StreamingABITBoundaryDetector(batch_size=batch_size, min_split_tokens=2, initial_threshold=0.5, window_size=5)
    
    # Pre-warm the detector
    dummy_hidden = np.random.randn(batch_size, head_dim)
    
    # Sequence 0 gets a duplicate (should NOT trigger semantic split initially)
    # Sequence 1 gets orthogonal (SHOULD trigger semantic split)
    seq0_step1 = np.array([1.0] * head_dim)
    seq1_step1 = np.array([1.0] * head_dim)
    
    seq0_step2 = seq0_step1  # Identical, won't split
    seq1_step2 = np.array([-1.0] * head_dim)  # Orthogonal, will split
    detector.step(np.stack([seq0_step1, seq1_step1], axis=0))
    res1 = detector.step(np.stack([seq0_step2, seq1_step2], axis=0))
    res2 = detector.step(np.stack([seq0_step2, seq1_step2], axis=0))

    assert res2[0] is None, "Sequence 0 should not have split"
    assert res2[1] is not None, "Sequence 1 should have split due to orthogonality"
    assert res2[1].total_tokens == 2

def test_sonic_cruncher_batch_dim():
    """
    Placeholder to test if SONIC cruncher handles batch dimensions correctly.
    """
    pass
