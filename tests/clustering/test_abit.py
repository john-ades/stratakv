import numpy as np
import pytest
from src.clustering.abit import ClusterSegment, StreamingABITBoundaryDetector

def test_cluster_segment_property():
    embeddings = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    counts = [1, 1, 1]
    segment = ClusterSegment(embeddings=embeddings, token_counts=counts, total_tokens=3, medoid_idx=1)
    
    assert np.array_equal(segment.medoid_embedding, embeddings[1])

def test_abit_detector_initialization():
    detector = StreamingABITBoundaryDetector(
        batch_size=1,
        window_size=5,
        min_split_tokens=4,
        max_split_tokens=100,
        split_tokens_tolerance=2,
        threshold_adjustment=0.05,
        initial_threshold=0.6
    )
    
    assert detector.window_size == 5
    assert detector.min_split_tokens == 4
    assert detector.max_split_tokens == 100
    assert detector.split_tokens_tolerance == 2
    assert detector.threshold_adjustment == 0.05
    assert np.isclose(detector.current_threshold[0], 0.6)
    assert len(detector.active_embeddings[0]) == 0

def test_step_accumulation():
    detector = StreamingABITBoundaryDetector(min_split_tokens=3, max_split_tokens=10)
    emb = np.array([1.0, 0.0])
    
    # First token, buffer empty -> none returned, token accumulated
    res1 = detector.step(emb, count=1)
    assert res1[0] is None
    assert len(detector.active_embeddings[0]) == 1
    assert detector.active_total_tokens[0] == 1
    
    # Second token, buffer not empty, total less than min_split -> none returned
    res2 = detector.step(emb, count=1)
    assert res2[0] is None
    assert len(detector.active_embeddings[0]) == 2
    assert detector.active_total_tokens[0] == 2

def test_step_force_split():
    # Force split happens before appending new token if total_tokens + count > max
    detector = StreamingABITBoundaryDetector(min_split_tokens=2, max_split_tokens=3, initial_threshold=-1.0, threshold_adjustment=0.1)
    emb = np.array([1.0, 0.0])
    
    # 1 token
    assert detector.step(emb, count=1)[0] is None
    # 2 tokens
    assert detector.step(emb, count=1)[0] is None
    # 3 tokens
    assert detector.step(emb, count=1)[0] is None
    
    # 4th token triggers force split before insertion (max_split = 3)
    res = detector.step(emb, count=1)
    
    assert res[0] is not None
    assert isinstance(res[0], ClusterSegment)
    assert res[0].total_tokens == 3
    assert len(res[0].embeddings) == 3
    # Check that threshold was adjusted up due to force split
    assert np.isclose(detector.current_threshold[0], -0.9)
    
    # Ensure new token was accumulated
    assert detector.active_total_tokens[0] == 1
    assert len(detector.active_embeddings[0]) == 1

def test_step_semantic_split():
    # Semantic split occurs when total >= min_split and sim < threshold
    detector = StreamingABITBoundaryDetector(
        min_split_tokens=2, 
        max_split_tokens=10, 
        initial_threshold=0.5, 
        threshold_adjustment=0.1,
        split_tokens_tolerance=5
    )
    
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([0.0, 1.0]) # Orthogonal
    
    assert detector.step(emb1, count=1)[0] is None
    # Sim of emb1 to cumulative context [1.0, 0.0] is 1.0 > 0.5, so no split here even if min=2, but min is 2, so it only checks >= 2.
    # Total tokens becomes 2, cumulative context becomes mean of active
    assert detector.step(emb1, count=1)[0] is None 
    
    # Next step comes with orthogonal embedding. sim = 0.0 < 0.5. total = 2 >= 2
    res = detector.step(emb2, count=1)
    
    assert res[0] is not None
    assert res[0].total_tokens == 2
    
    # Split tokens = 2 <= (min=2 + tol=5) -> threshold is relaxed (decreased)
    assert np.isclose(detector.current_threshold[0], 0.4)
    
    assert detector.active_total_tokens[0] == 1
    assert np.array_equal(detector.active_embeddings[0][0], emb2)

def test_medoid_extraction():
    detector = StreamingABITBoundaryDetector()
    
    emb1 = np.array([1.0, 0.0])
    emb2 = np.array([np.sqrt(0.5), np.sqrt(0.5)]) # More central between emb1 and emb3
    emb3 = np.array([0.0, 1.0])
    
    detector.step(emb1, count=1)
    detector.step(emb2, count=1)
    detector.step(emb3, count=1)
    
    segment = detector._seal_cluster(0)
    
    assert hasattr(segment, 'medoid_idx')
    assert segment.medoid_idx == 1  # emb2 is the medoid

def test_medoid_extraction_single():
    detector = StreamingABITBoundaryDetector()
    detector.step(np.array([1.0, 0.0]), count=1)
    segment = detector._seal_cluster(0)
    assert segment.medoid_idx == 0

def test_flush_empty():
    detector = StreamingABITBoundaryDetector()
    assert detector.flush()[0] is None

def test_flush_data():
    detector = StreamingABITBoundaryDetector()
    detector.step(np.array([1.0, 0.0]))
    res = detector.flush()[0]
    assert res is not None
    assert res.total_tokens == 1
    assert len(detector.active_embeddings[0]) == 0
    assert detector.active_total_tokens[0] == 0

def test_process_batch():
    detector = StreamingABITBoundaryDetector(batch_size=1, min_split_tokens=2, max_split_tokens=4, initial_threshold=0.5)
    
    X = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0], # Orthogonal -> triggers split with preceding tokens (len=3 >= min=2)
        [0.0, 1.0]
    ])
    T = np.array([1, 1, 1, 1, 1])
    
    segments = detector.process_batch(X, T)
    
    # segments is a list for batch_size (length 1). segments[0] is the list of clusters for batch index 0.
    assert len(segments[0]) == 1
    assert segments[0][0].total_tokens == 3
    assert np.array_equal(segments[0][0].embeddings[0], [1.0, 0.0])
    
    # Remaining 2 tokens are in buffer
    assert detector.active_total_tokens[0] == 2
    
    # Flush remaining
    final = detector.flush()[0]
    assert final.total_tokens == 2
    assert np.array_equal(final.embeddings[0], [0.0, 1.0])
