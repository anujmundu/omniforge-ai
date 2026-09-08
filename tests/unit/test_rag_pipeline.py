"""
Unit tests for End-to-End Enterprise RAG Pipeline (Ingestion -> Retrieval -> Generation).
"""

import pytest
from rag.base import Document
from rag.pipeline import EnterpriseRAGPipeline


@pytest.fixture
def rag_pipeline():
    pipeline = EnterpriseRAGPipeline()
    docs = [
        Document(
            title="OmniForge Overview",
            content=(
                "OmniForge is a production-grade multimodal AI/ML platform. "
                "It integrates Classical ML, Computer Vision with YOLO and ByteTrack, "
                "NLP pipelines with Transformer embeddings and NER, and an Enterprise RAG Engine."
            ),
            source_type="text",
        ),
        Document(
            title="Vector Database Guide",
            content=(
                "Dense vector embeddings represent unstructured text in high-dimensional semantic spaces. "
                "Cosine similarity measures the angular distance between vectors to retrieve relevant passages."
            ),
            source_type="text",
        ),
    ]
    pipeline.index_documents("kb_test", docs)
    return pipeline


def test_rag_pipeline_retrieval(rag_pipeline):
    results = rag_pipeline.retrieve(
        query="What modules are supported in OmniForge multimodal platform?",
        collection_name="kb_test",
        top_k=2,
        rerank=True,
    )
    assert len(results) >= 1
    assert "OmniForge" in results[0].chunk.title
    assert results[0].rerank_score is not None


def test_rag_pipeline_query_with_citations(rag_pipeline):
    res = rag_pipeline.query(
        query="Explain vector embeddings and cosine similarity",
        collection_name="kb_test",
        top_k=2,
        rerank=True,
    )

    assert res.query == "Explain vector embeddings and cosine similarity"
    assert len(res.citations) >= 1
    assert res.citations[0].doc_title == "Vector Database Guide"
    assert "[1]" in res.answer
    assert res.latency_ms > 0.0


def test_rag_pipeline_empty_collection(rag_pipeline):
    res = rag_pipeline.query(
        query="Non-existent information query",
        collection_name="empty_coll",
        top_k=2,
    )

    assert "No relevant documentation was found" in res.answer
    assert len(res.citations) == 0
