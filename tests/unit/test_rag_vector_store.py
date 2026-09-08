"""
Unit tests for InMemoryVectorStore collections, dense indexing, and cosine search.
"""

import pytest
from rag.base import DocumentChunk
from rag.vector_store import InMemoryVectorStore


@pytest.fixture
def vector_store():
    return InMemoryVectorStore()


def test_add_and_search_chunks(vector_store):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            title="Vision Guide",
            text="YOLO object detection and ByteTrack multi-object tracking are implemented in Computer Vision.",
            chunk_index=0,
            start_char=0,
            end_char=80,
            metadata={"domain": "vision"},
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            title="NLP Guide",
            text="Transformer embeddings and Named Entity Recognition are part of the NLP pipeline.",
            chunk_index=0,
            start_char=0,
            end_char=80,
            metadata={"domain": "nlp"},
        ),
        DocumentChunk(
            chunk_id="c3",
            doc_id="d3",
            title="Finance Report",
            text="Q3 revenue grew by 24% year-over-year with strong margins.",
            chunk_index=0,
            start_char=0,
            end_char=60,
            metadata={"domain": "finance"},
        ),
    ]

    added = vector_store.add_chunks("tech_kb", chunks)
    assert added == 3

    # Query with NLP query
    q_emb = vector_store.embedder.embed_text("transformer entity recognition nlp")
    results = vector_store.search("tech_kb", query_vector=q_emb.vector, top_k=2)

    assert len(results) == 2
    assert results[0].chunk.doc_id == "d2"
    assert results[0].similarity_score > 0.4
    assert results[0].rank == 1


def test_metadata_filtering(vector_store):
    chunks = [
        DocumentChunk(
            chunk_id="c1",
            doc_id="d1",
            title="Doc 1",
            text="Machine learning algorithms and optimization methods.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata={"env": "prod"},
        ),
        DocumentChunk(
            chunk_id="c2",
            doc_id="d2",
            title="Doc 2",
            text="Machine learning pipelines and continuous integration.",
            chunk_index=0,
            start_char=0,
            end_char=50,
            metadata={"env": "staging"},
        ),
    ]

    vector_store.add_chunks("test_kb", chunks)
    q_emb = vector_store.embedder.embed_text("machine learning")

    results_prod = vector_store.search(
        "test_kb",
        query_vector=q_emb.vector,
        top_k=5,
        metadata_filter={"env": "prod"},
    )
    assert len(results_prod) == 1
    assert results_prod[0].chunk.metadata["env"] == "prod"


def test_list_collections_and_clear(vector_store):
    chunks = [
        DocumentChunk(chunk_id="c1", doc_id="d1", title="T", text="Sample text", chunk_index=0, start_char=0, end_char=11)
    ]
    vector_store.add_chunks("coll_a", chunks)
    vector_store.add_chunks("coll_b", chunks)

    colls = vector_store.list_collections()
    assert len(colls) == 2
    names = [c["collection_name"] for c in colls]
    assert "coll_a" in names
    assert "coll_b" in names

    vector_store.clear("coll_a")
    colls_after = vector_store.list_collections()
    assert len(colls_after) == 1
    assert colls_after[0]["collection_name"] == "coll_b"
