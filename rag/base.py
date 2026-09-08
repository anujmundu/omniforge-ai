"""
Base contracts and domain models for Enterprise Retrieval-Augmented Generation (RAG).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


import uuid

class Document(BaseModel):
    """Raw or structured document representation before chunking."""
    doc_id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4().hex[:12]}", description="Unique document identifier")
    title: str = Field(..., description="Document title or filename")
    content: str = Field(..., description="Raw text content")
    source_type: str = Field(default="text", description="Document source format (text, markdown, json, pdf)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Custom document metadata")


class DocumentChunk(BaseModel):
    """Segmented semantic chunk of a parent document."""
    chunk_id: str = Field(..., description="Unique chunk ID (e.g. {doc_id}_{index})")
    doc_id: str = Field(..., description="Parent document identifier")
    title: str = Field(..., description="Parent document title")
    text: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., ge=0, description="0-indexed position in parent document")
    start_char: int = Field(default=0, ge=0, description="Start character offset in parent content")
    end_char: int = Field(default=0, ge=0, description="End character offset in parent content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata propagated from parent doc")
    vector: Optional[List[float]] = Field(default=None, description="Cached embedding vector if indexed")


class RetrievalResult(BaseModel):
    """Retrieved document chunk with similarity / reranking scores."""
    chunk: DocumentChunk
    similarity_score: float = Field(..., description="Initial vector cosine similarity score")
    rerank_score: Optional[float] = Field(default=None, description="Second-stage cross-encoder score")
    rank: int = Field(default=1, ge=1, description="Rank position (1-indexed)")


class Citation(BaseModel):
    """Grounded source citation for generated answer."""
    citation_id: int = Field(..., description="Citation marker (e.g. [1], [2])")
    doc_id: str
    doc_title: str
    chunk_id: str
    chunk_index: int
    snippet: str = Field(..., description="Supporting text snippet")
    relevance_score: float


class RAGResponse(BaseModel):
    """Complete grounded answer synthesis output."""
    query: str
    answer: str = Field(..., description="Generated grounded response with citation markers")
    citations: List[Citation] = Field(default_factory=list, description="Verified source citations")
    retrieved_chunks: List[RetrievalResult] = Field(default_factory=list, description="Retrieved candidate chunks")
    model_name: str = Field(default="omniforge-rag-v1", description="Generator / pipeline identifier")
    latency_ms: float = Field(default=0.0, description="Total pipeline latency in milliseconds")


class RAGEvaluationResult(BaseModel):
    """Quantitative RAG evaluation metrics."""
    query: str
    generated_answer: str
    ground_truth_answer: Optional[str] = None
    faithfulness_score: float = Field(..., ge=0.0, le=1.0, description="Groundedness in retrieved context (0-1)")
    answer_relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance to user question (0-1)")
    context_precision_score: float = Field(..., ge=0.0, le=1.0, description="Precision of retrieved chunks (0-1)")
    overall_rag_score: float = Field(..., ge=0.0, le=1.0, description="Harmonic mean of evaluation metrics")
    evaluation_details: Dict[str, Any] = Field(default_factory=dict)


# ==============================================================================
# Abstract Engine Interfaces
# ==============================================================================

class BaseDocumentParser(ABC):
    """Abstract interface for parsing heterogeneous document formats."""

    @abstractmethod
    def parse(self, raw_content: str, title: str, source_type: str = "text", metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Parse raw content into a standardized Document instance."""
        pass


class BaseChunker(ABC):
    """Abstract interface for splitting documents into semantic chunks."""

    @abstractmethod
    def split_document(self, document: Document) -> List[DocumentChunk]:
        """Decompose a Document into structured chunks."""
        pass


class BaseVectorStore(ABC):
    """Abstract interface for dense vector indexing and retrieval."""

    @abstractmethod
    def add_chunks(self, collection_name: str, chunks: List[DocumentChunk]) -> int:
        """Index chunks and their vectors into a named collection."""
        pass

    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], top_k: int = 5) -> List[RetrievalResult]:
        """Perform dense vector cosine search in a named collection."""
        pass

    @abstractmethod
    def list_collections(self) -> List[Dict[str, Any]]:
        """List active vector collections and chunk statistics."""
        pass


class BaseReranker(ABC):
    """Abstract interface for cross-encoder reranking."""

    @abstractmethod
    def rerank(self, query: str, candidate_chunks: List[RetrievalResult], top_k: int = 3) -> List[RetrievalResult]:
        """Re-score and re-order candidate chunks using contextual cross-attention."""
        pass


class BaseRAGPipeline(ABC):
    """Abstract interface for end-to-end RAG question answering."""

    @abstractmethod
    def query(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> RAGResponse:
        """Execute grounded retrieval and answer synthesis."""
        pass
