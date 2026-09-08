"""
Unit tests for RAGEvaluator metrics (Faithfulness, Relevance, Context Precision).
"""

import pytest
from rag.base import DocumentChunk, RetrievalResult
from rag.evaluator import RAGEvaluator


def test_rag_evaluator_metrics():
    evaluator = RAGEvaluator()

    query = "What are the core capabilities of OmniForge?"
    answer = "OmniForge provides Classical ML, Computer Vision, and NLP pipelines [1]."
    retrieved = [
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="c1",
                doc_id="d1",
                title="OmniForge Overview",
                text="OmniForge provides Classical ML, Computer Vision, and NLP pipelines for enterprise AI systems.",
                chunk_index=0,
                start_char=0,
                end_char=90,
            ),
            similarity_score=0.92,
            rerank_score=0.95,
            rank=1,
        )
    ]

    res = evaluator.evaluate(
        query=query,
        generated_answer=answer,
        retrieved_chunks=retrieved,
        ground_truth_answer="OmniForge supports Classical ML, Computer Vision, and NLP.",
    )

    assert res.faithfulness_score >= 0.70
    assert res.answer_relevance_score >= 0.70
    assert res.context_precision_score >= 0.85
    assert res.overall_rag_score >= 0.70
    assert res.evaluation_details["retrieved_chunk_count"] == 1
