"""
NLP API Router for Embeddings, Named Entity Recognition, Classification, and Semantic Similarity.
"""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Depends, status

from apps.api.core.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.schemas.nlp import (
    ClassificationPredictionSchema,
    ClassifyRequest,
    ClassifyResponse,
    EmbedRequest,
    EmbedResponse,
    NamedEntitySpanSchema,
    NERRequest,
    NERResponse,
    NLPModelInfoResponse,
    RankedDocumentSchema,
    SimilarityRequest,
    SimilarityResponse,
    TextEmbeddingItemSchema,
)
from nlp.classification import TextClassifier
from nlp.embeddings import TransformerEmbeddingEngine
from nlp.ner import NamedEntityRecognizer
from nlp.similarity import SemanticSimilarityEngine

router = APIRouter(prefix="/nlp", tags=["Natural Language Processing (NLP)"])

# In-memory singleton instances for fast inference
_embedder = TransformerEmbeddingEngine(model_name="all-MiniLM-L6-v2", dimension=384)
_ner_engine = NamedEntityRecognizer()
_classifier = TextClassifier()
_similarity_engine = SemanticSimilarityEngine(embedding_engine=_embedder)


@router.get("/models", response_model=NLPModelInfoResponse)
async def get_nlp_models(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    List available NLP models, vector dimensionality, and entity types.
    """
    return NLPModelInfoResponse(
        models=[
            {
                "model_id": "all-MiniLM-L6-v2",
                "task": "dense_text_embeddings",
                "dimensions": 384,
                "pooling": "mean / CLS",
                "normalization": "L2 unit norm",
            },
            {
                "model_id": "spacy_transformer_ner_hybrid",
                "task": "named_entity_recognition",
                "span_alignment": "exact character indices",
            },
            {
                "model_id": "distilbert_sentiment_classifier",
                "task": "text_classification_and_sentiment",
                "probability_model": "Softmax",
            },
            {
                "model_id": "cosine_semantic_search",
                "task": "semantic_similarity_and_ranking",
                "metric": "Cosine Similarity",
            },
        ],
        entity_types=["PERSON", "ORG", "GPE", "MONEY", "DATE", "TECH_STACK", "PRODUCT"],
        default_classes=["POSITIVE", "NEUTRAL", "NEGATIVE"],
    )


@router.post("/embed", response_model=EmbedResponse)
async def generate_embeddings(
    request: EmbedRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Generate dense vector embeddings for input strings.
    """
    batch_res = _embedder.embed_batch(request.texts)
    
    embeddings = [
        TextEmbeddingItemSchema(
            text=e.text,
            vector=e.vector,
            dimension=e.dimension,
            model_name=e.model_name,
            normalized=e.normalized,
        )
        for e in batch_res.embeddings
    ]

    return EmbedResponse(
        total_embeddings=len(embeddings),
        total_tokens=batch_res.total_tokens,
        dimension=batch_res.dimension,
        embeddings=embeddings,
        inference_latency_ms=batch_res.inference_latency_ms,
    )


@router.post("/ner", response_model=NERResponse)
async def extract_named_entities(
    request: NERRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Extract localized named entity spans with verified character offsets.
    """
    ner_result = _ner_engine.extract_entities(
        text=request.text,
        min_confidence=request.min_confidence,
    )

    entities = [
        NamedEntitySpanSchema(
            text=e.text,
            label=e.label,
            start_char=e.start_char,
            end_char=e.end_char,
            confidence=e.confidence,
        )
        for e in ner_result.entities
    ]

    return NERResponse(
        source_text=ner_result.source_text,
        total_entities=len(entities),
        entities=entities,
        inference_latency_ms=ner_result.inference_latency_ms,
    )


@router.post("/classify", response_model=ClassifyResponse)
async def classify_text(
    request: ClassifyRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Predict normalized class probability distributions for input text.
    """
    result = _classifier.classify(
        text=request.text,
        candidate_labels=request.candidate_labels,
    )

    probs = [
        ClassificationPredictionSchema(label=p.label, score=p.score)
        for p in result.probabilities
    ]

    return ClassifyResponse(
        source_text=result.source_text,
        top_label=result.top_label,
        top_score=result.top_score,
        probabilities=probs,
        inference_latency_ms=result.inference_latency_ms,
    )


@router.post("/similarity", response_model=SimilarityResponse)
async def compute_semantic_similarity(
    request: SimilarityRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Compute pairwise cosine similarity matrix and retrieve top-K ranked documents.
    """
    if request.query:
        result = _similarity_engine.search_top_k(
            query=request.query,
            documents=request.documents,
            top_k=request.top_k,
        )
    else:
        result = _similarity_engine.compute_similarity_matrix(
            texts=request.documents,
        )

    top_matches = (
        [
            RankedDocumentSchema(
                document_index=m.document_index,
                text=m.text,
                similarity_score=m.similarity_score,
            )
            for m in result.top_k_matches
        ]
        if result.top_k_matches
        else None
    )

    return SimilarityResponse(
        query=result.query_text,
        total_documents=len(request.documents),
        similarity_matrix=result.similarity_matrix,
        top_k_matches=top_matches,
        inference_latency_ms=result.inference_latency_ms,
    )
