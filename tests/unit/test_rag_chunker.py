"""
Unit tests for Recursive Semantic Chunking and Exact Span Alignment.
"""

from rag.base import Document
from rag.chunker import RecursiveSemanticChunker


def test_chunker_basic_splitting():
    chunker = RecursiveSemanticChunker(chunk_size=150, chunk_overlap=30)
    text = (
        "OmniForge is an enterprise AI/ML intelligence platform. "
        "It provides modular engines for Classical Machine Learning, Computer Vision, and NLP. "
        "The RAG engine supports dense vector search, hybrid retrieval, and cross-encoder reranking. "
        "With verifiable source citations, users get fully grounded responses."
    )
    doc = Document(title="Platform Overview", content=text, source_type="text")
    chunks = chunker.split_document(doc)

    assert len(chunks) >= 2
    for idx, chunk in enumerate(chunks):
        assert chunk.doc_id == doc.doc_id
        assert chunk.title == doc.title
        assert chunk.chunk_index == idx
        # Verify exact substring match with character offsets
        assert text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunker_short_text():
    chunker = RecursiveSemanticChunker(chunk_size=500, chunk_overlap=50)
    doc = Document(title="Short", content="Short document text.", source_type="text")
    chunks = chunker.split_document(doc)

    assert len(chunks) == 1
    assert chunks[0].text == "Short document text."
    assert chunks[0].start_char == 0
    assert chunks[0].end_char == len("Short document text.")


def test_chunker_empty_document():
    chunker = RecursiveSemanticChunker()
    doc = Document(title="Empty", content="   ", source_type="text")
    chunks = chunker.split_document(doc)
    assert len(chunks) == 0
