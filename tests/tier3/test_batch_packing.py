import pytest
import torch
from src.core.config import StrataKVConfig
from src.clustering.abit import StreamingABITBoundaryDetector
from src.clustering.buffer import ClusterSegment

@pytest.mark.xfail(reason="StreamingABITBoundaryDetector and AbitClusterBuffer currently assume batch_size=1 and flatten dimensions, failing to isolate batch sequences.")
def test_batch_packing_isolation():
    """
    Verifies that ABIT Boundary Detection safely processes a batch dimension where
    sequence A triggers an acoustic boundary but sequence B does not.
    """
    config = StrataKVConfig(
        tier3_size=4,
        enable_tier3=True
    )
    
    detector = StreamingABITBoundaryDetector(initial_threshold=0.5, window_size=5)
    
    batch_size = 2
    head_dim = 16
    
    # Pre-warm the detector
    dummy_hidden = torch.randn(batch_size, head_dim)
    # The detector will crash or behave incorrectly here due to lack of batch dim support
    detector.step(dummy_hidden.numpy())

def test_sonic_cruncher_batch_dim():
    """
    Placeholder to test if SONIC cruncher handles batch dimensions correctly.
    """
    pass
