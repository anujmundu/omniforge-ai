"""
Unit tests for Transformer text embeddings.
"""

import numpy as np
import pytest
from nlp.embeddings import TransformerEmbeddingEngine


def test_embedding_dimensionality_and_l2_norm():
    engine = TransformerEmbeddingEngine(dimension=384)
    text = "OmniForge AI is an advanced multimodal intelligence platform."

    emb = engine.embed_text(text)
    assert emb.dimension == 384
    assert len(emb.vector) == 384

    # Check L2 unit normalization: ||v|| = 1.0
    vec_arr = np.array(emb.vector, dtype=np.float32)
    norm = np.linalg.norm(vec_arr)
    assert norm == pytest.approx(1.0, rel=1e-4)


def test_batch_embedding_generation():
    engine = TransformerEmbeddingEngine(dimension=128)
    texts = [
        "Python FastAPI backend",
        "Deep neural networks and PyTorch",
        "Quarterly fiscal financial earnings report"
    ]

    batch = engine.embed_batch(texts)
    assert len(batch.embeddings) == 3
    assert batch.total_tokens > 0
    assert batch.dimension == 128
    assert batch.inference_latency_ms >= 0.0


def test_cosine_similarity_calculation():
    engine = TransformerEmbeddingEngine(dimension=384)

    t1 = "Machine learning model training with PyTorch and GPU"
    t2 = "Deep learning neural network training on GPU clusters"
    t3 = "Corporate financial revenue profit dividends and quarterly balance sheet"

    emb1 = engine.embed_text(t1)
    emb2 = engine.embed_text(t2)
    emb3 = engine.embed_text(t3)

    sim_1_2 = engine.cosine_similarity(emb1.vector, emb2.vector)
    sim_1_3 = engine.cosine_similarity(emb1.vector, emb3.vector)

    # ML topics should have higher similarity than ML vs Financial balance sheet
    assert sim_1_2 > sim_1_3
