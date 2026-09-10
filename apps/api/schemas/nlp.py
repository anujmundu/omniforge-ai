"""
Pydantic schemas for NLP REST APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TextEmbeddingItemSchema(BaseModel):
    text: str
    vector: List[float]
    dimension: int
    model_name: str
    normalized: bool


class EmbedRequest(BaseModel):
    """Request payload for text embedding generation."""

    texts: List[str] = Field(..., min_length=1, description="List of strings to embed")
    dimension: Optional[int] = Field(default=384, description="Target vector dimension")


class EmbedResponse(BaseModel):
    """Response payload for text embedding generation."""

    total_embeddings: int
    total_tokens: int
    dimension: int
    embeddings: List[TextEmbeddingItemSchema]
    inference_latency_ms: float


class NamedEntitySpanSchema(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int
    confidence: float


class NERRequest(BaseModel):
    """Request payload for Named Entity Recognition."""

    text: str = Field(..., min_length=1, description="Source text to extract entities from")
    min_confidence: float = Field(default=0.50, ge=0.0, le=1.0)


class NERResponse(BaseModel):
    """Response payload for Named Entity Recognition."""

    source_text: str
    total_entities: int
    entities: List[NamedEntitySpanSchema]
    inference_latency_ms: float


class ClassificationPredictionSchema(BaseModel):
    label: str
    score: float


class ClassifyRequest(BaseModel):
    """Request payload for text classification / sentiment analysis."""

    text: str = Field(..., min_length=1)
    candidate_labels: Optional[List[str]] = Field(
        default=None, description="Optional candidate labels (defaults to POSITIVE, NEUTRAL, NEGATIVE)"
    )


class ClassifyResponse(BaseModel):
    """Response payload for text classification."""

    source_text: str
    top_label: str
    top_score: float
    probabilities: List[ClassificationPredictionSchema]
    inference_latency_ms: float


class SimilarityRequest(BaseModel):
    """Request payload for cross-document similarity & search."""

    query: Optional[str] = Field(default=None, description="Optional search query text")
    documents: List[str] = Field(..., min_length=1, description="Candidate document strings")
    top_k: int = Field(default=5, ge=1)


class RankedDocumentSchema(BaseModel):
    document_index: int
    text: str
    similarity_score: float


class SimilarityResponse(BaseModel):
    """Response payload for document similarity."""

    query: Optional[str] = None
    total_documents: int
    similarity_matrix: List[List[float]]
    top_k_matches: Optional[List[RankedDocumentSchema]] = None
    inference_latency_ms: float


class NLPModelInfoResponse(BaseModel):
    models: List[Dict[str, Any]]
    entity_types: List[str]
    default_classes: List[str]
