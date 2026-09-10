"""
Unit tests for Cross-Encoder Reranking and Semantic Relevance Scoring.
"""

from rag.base import DocumentChunk, RetrievalResult
from rag.reranker import CrossEncoderReranker


def test_cross_encoder_rerank():
    reranker = CrossEncoderReranker()

    candidates = [
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="c1",
                doc_id="d1",
                title="Cloud Architecture",
                text="Kubernetes cluster orchestration with auto-scaling pods in AWS.",
                chunk_index=0,
                start_char=0,
                end_char=60,
            ),
            similarity_score=0.75,
            rank=1,
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id="c2",
                doc_id="d2",
                title="RAG Retrival",
                text="Cross-encoder reranking significantly boosts precision in Enterprise RAG retrieval pipelines.",
                chunk_index=0,
                start_char=0,
                end_char=90,
            ),
            similarity_score=0.70,
            rank=2,
        ),
    ]

    query = "How does cross-encoder reranking improve RAG retrieval?"
    reranked = reranker.rerank(query=query, candidate_chunks=candidates, top_k=2)

    assert len(reranked) == 2
    # Document 2 should be boosted to rank 1 due to exact conceptual cross-attention match
    assert reranked[0].chunk.doc_id == "d2"
    assert reranked[0].rerank_score is not None
    assert reranked[0].rerank_score > reranked[1].rerank_score
    assert reranked[0].rank == 1
    assert reranked[1].rank == 2
