"""
Base contracts and domain models for Natural Language Processing (NLP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from pydantic import BaseModel, Field


class TextEmbedding(BaseModel):
    """Dense vector representation of a text sequence."""

    text: str = Field(..., description="Original input text")
    vector: List[float] = Field(..., description="Dense embedding vector")
    dimension: int = Field(..., description="Dimensionality of the vector space")
    model_name: str = Field(default="all-MiniLM-L6-v2", description="Embedding model identifier")
    normalized: bool = Field(default=True, description="Whether the vector is L2 unit normalized")


class BatchEmbeddingResult(BaseModel):
    """Batch of text embeddings with aggregate statistics."""

    embeddings: List[TextEmbedding] = Field(default_factory=list)
    total_tokens: int = Field(default=0, description="Estimated total tokens processed")
    dimension: int = Field(..., description="Dimensionality of embedding vectors")
    inference_latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")


class NamedEntitySpan(BaseModel):
    """Named entity mention with exact character offsets in source text."""

    text: str = Field(..., description="Entity surface text substring")
    label: str = Field(..., description="Entity category label (e.g. PERSON, ORG, GPE, MONEY, TECH_STACK)")
    start_char: int = Field(..., ge=0, description="0-indexed start character offset")
    end_char: int = Field(..., ge=0, description="0-indexed exclusive end character offset")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class NERResult(BaseModel):
    """Complete Named Entity Recognition output for a document."""

    source_text: str = Field(..., description="Source text analyzed")
    entities: List[NamedEntitySpan] = Field(default_factory=list, description="Extracted entity spans")
    total_entities: int = Field(default=0, description="Number of detected entities")
    inference_latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")


class ClassificationPrediction(BaseModel):
    """Single class label probability prediction."""

    label: str = Field(..., description="Predicted class label")
    score: float = Field(..., ge=0.0, le=1.0, description="Probability / confidence score")


class ClassificationResult(BaseModel):
    """Text classification & sentiment analysis result."""

    source_text: str = Field(..., description="Input text classified")
    top_label: str = Field(..., description="Highest probability predicted class label")
    top_score: float = Field(..., ge=0.0, le=1.0, description="Highest probability score")
    probabilities: List[ClassificationPrediction] = Field(
        default_factory=list, description="Full probability distribution over all classes"
    )
    inference_latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")


class RankedDocument(BaseModel):
    """Document ranked by semantic similarity score."""

    document_index: int = Field(..., description="Index of document in candidate list")
    text: str = Field(..., description="Candidate document text")
    similarity_score: float = Field(..., ge=-1.0, le=1.0, description="Cosine similarity score")


class SimilarityMatrixResult(BaseModel):
    """Pairwise semantic similarity matrix and top-K ranked matches."""

    query_text: Optional[str] = Field(default=None, description="Query text if asymmetric search")
    documents: List[str] = Field(default_factory=list, description="Candidate document list")
    similarity_matrix: List[List[float]] = Field(
        default_factory=list, description="N x N or 1 x N pairwise cosine similarity values"
    )
    top_k_matches: Optional[List[RankedDocument]] = Field(
        default=None, description="Top-K nearest documents sorted descending by score"
    )
    inference_latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")


# ==============================================================================
# Abstract Engine Interfaces
# ==============================================================================


class BaseEmbeddingModel(ABC):
    """Abstract interface for text embedding engines."""

    @abstractmethod
    def embed_text(self, text: str) -> TextEmbedding:
        """Generate dense embedding vector for a single string."""
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> BatchEmbeddingResult:
        """Generate dense embeddings for a batch of strings."""
        pass


class BaseNERModel(ABC):
    """Abstract interface for Named Entity Recognition engines."""

    @abstractmethod
    def extract_entities(self, text: str, min_confidence: float = 0.50) -> NERResult:
        """Extract localized entity spans with character indices."""
        pass


class BaseTextClassifier(ABC):
    """Abstract interface for text classification and sentiment engines."""

    @abstractmethod
    def classify(self, text: str, candidate_labels: Optional[List[str]] = None) -> ClassificationResult:
        """Predict class probabilities for input text."""
        pass


class BaseSimilarityEngine(ABC):
    """Abstract interface for cross-document semantic similarity ranking."""

    @abstractmethod
    def compute_similarity_matrix(self, texts: List[str]) -> SimilarityMatrixResult:
        """Compute pairwise cosine similarity matrix for a list of texts."""
        pass

    @abstractmethod
    def search_top_k(self, query: str, documents: List[str], top_k: int = 5) -> SimilarityMatrixResult:
        """Retrieve top-K most semantically similar documents to a query string."""
        pass
