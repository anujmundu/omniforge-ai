"""
OmniForge Natural Language Processing (NLP) Package.
"""

from nlp.base import (
    BaseEmbeddingModel,
    BaseNERModel,
    BaseSimilarityEngine,
    BaseTextClassifier,
    BatchEmbeddingResult,
    ClassificationPrediction,
    ClassificationResult,
    NamedEntitySpan,
    NERResult,
    RankedDocument,
    SimilarityMatrixResult,
    TextEmbedding,
)
from nlp.classification import TextClassifier
from nlp.embeddings import TransformerEmbeddingEngine
from nlp.ner import NamedEntityRecognizer
from nlp.similarity import SemanticSimilarityEngine

__all__ = [
    "BaseEmbeddingModel",
    "BaseNERModel",
    "BaseTextClassifier",
    "BaseSimilarityEngine",
    "TextEmbedding",
    "BatchEmbeddingResult",
    "NamedEntitySpan",
    "NERResult",
    "ClassificationPrediction",
    "ClassificationResult",
    "RankedDocument",
    "SimilarityMatrixResult",
    "TransformerEmbeddingEngine",
    "NamedEntityRecognizer",
    "TextClassifier",
    "SemanticSimilarityEngine",
]
