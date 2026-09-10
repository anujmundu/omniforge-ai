"""
Unit tests for semantic similarity engine and top-K search.
"""

import pytest

from nlp.similarity import SemanticSimilarityEngine


def test_similarity_matrix_symmetry_and_identity():
    engine = SemanticSimilarityEngine()
    docs = [
        "Distributed machine learning systems with GPU acceleration",
        "Deep neural network training with PyTorch clusters",
        "Quarterly fiscal financial revenue reports and profit margins",
    ]

    res = engine.compute_similarity_matrix(docs)
    mat = res.similarity_matrix

    assert len(mat) == 3
    assert len(mat[0]) == 3

    # Check diagonal identity (M[i][i] == 1.0)
    for i in range(3):
        assert mat[i][i] == pytest.approx(1.0, rel=1e-3)

    # Check symmetry (M[i][j] == M[j][i])
    for i in range(3):
        for j in range(3):
            assert mat[i][j] == pytest.approx(mat[j][i], rel=1e-3)


def test_search_top_k_relevance_ranking():
    engine = SemanticSimilarityEngine()
    docs = [
        "Python FastAPI web development framework",
        "Corporate annual financial balance sheet",
        "Machine learning model training and inference with PyTorch",
        "Customer support ticket management",
    ]

    query = "How to train deep learning models using GPUs and PyTorch"
    res = engine.search_top_k(query=query, documents=docs, top_k=2)

    assert len(res.top_k_matches) == 2
    # The most relevant document must be the ML document (index 2)
    assert res.top_k_matches[0].document_index == 2
    assert "Machine learning" in res.top_k_matches[0].text
    assert res.top_k_matches[0].similarity_score > res.top_k_matches[1].similarity_score
