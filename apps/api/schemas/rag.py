"""
Pydantic schemas for Enterprise RAG REST APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentInputSchema(BaseModel):
    title: str = Field(..., description="Document title or filename")
    content: str = Field(..., min_length=1, description="Raw text or formatted content")
    source_type: str = Field(default="text", description="text | markdown | json")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    doc_id: Optional[str] = None


class IndexDocumentsRequest(BaseModel):
    collection_name: str = Field(default="default_kb", min_length=1)
    documents: List[DocumentInputSchema] = Field(..., min_length=1)


class IndexDocumentsResponse(BaseModel):
    collection_name: str
    total_documents: int
    total_chunks_created: int
    dimension: int
    latency_ms: float


class DocumentChunkSchema(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]


class RetrievalItemSchema(BaseModel):
    chunk: DocumentChunkSchema
    similarity_score: float
    rerank_score: Optional[float] = None
    rank: int


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_name: str = Field(default="default_kb")
    top_k: int = Field(default=5, ge=1)
    rerank: bool = Field(default=True)


class RetrieveResponse(BaseModel):
    query: str
    collection_name: str
    total_retrieved: int
    results: List[RetrievalItemSchema]
    latency_ms: float


class CitationSchema(BaseModel):
    citation_id: int
    doc_id: str
    doc_title: str
    chunk_id: str
    chunk_index: int
    snippet: str
    relevance_score: float


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    collection_name: str = Field(default="default_kb")
    top_k: int = Field(default=3, ge=1)
    rerank: bool = Field(default=True)


class RAGQueryResponse(BaseModel):
    query: str
    answer: str
    citations: List[CitationSchema]
    retrieved_chunks: List[RetrievalItemSchema]
    model_name: str
    latency_ms: float


class RAGEvaluateRequest(BaseModel):
    query: str = Field(..., min_length=1)
    generated_answer: str = Field(..., min_length=1)
    retrieved_chunk_texts: List[str] = Field(..., min_length=1)
    ground_truth_answer: Optional[str] = None


class RAGEvaluateResponse(BaseModel):
    query: str
    faithfulness_score: float
    answer_relevance_score: float
    context_precision_score: float
    overall_rag_score: float
    evaluation_details: Dict[str, Any]


class CollectionInfoSchema(BaseModel):
    collection_name: str
    total_chunks: int
    unique_documents: int
    vector_dimension: int


class ListCollectionsResponse(BaseModel):
    collections: List[CollectionInfoSchema]
    total_collections: int
