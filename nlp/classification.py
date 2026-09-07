"""
Text Classification and Sentiment Analysis Engine with Softmax probability distributions.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional
import numpy as np

from nlp.base import BaseTextClassifier, ClassificationPrediction, ClassificationResult


class TextClassifier(BaseTextClassifier):
    """
    Production Text Classification and Sentiment Analysis engine.
    Computes normalized probability distributions over candidate target classes.
    """

    DEFAULT_CLASSES = ["POSITIVE", "NEUTRAL", "NEGATIVE"]

    LEXICON_WEIGHTS: Dict[str, Dict[str, float]] = {
        # Sentiment keywords
        "POSITIVE": {"great": 2.0, "excellent": 2.5, "fast": 1.5, "love": 2.0, "clean": 1.2, "production": 1.0, "success": 2.2, "robust": 1.8, "reliable": 2.2, "exceptionally": 1.5, "well-designed": 2.0, "designed": 1.0, "amazing": 2.5, "best": 2.0},
        "NEGATIVE": {"error": 2.5, "bug": 2.0, "fail": 2.2, "slow": 1.8, "broken": 2.5, "crash": 3.0, "terrible": 2.8, "issue": 1.5, "poor": 2.0, "bad": 2.2},
        "NEUTRAL": {"report": 1.0, "system": 1.0, "data": 1.0, "file": 1.0, "request": 1.0, "endpoint": 1.0, "process": 1.0, "standard": 1.0},
        # Topic keywords
        "TECHNOLOGY": {"code": 2.0, "api": 2.0, "python": 2.5, "docker": 2.5, "database": 2.0, "model": 1.8, "gpu": 2.0},
        "FINANCE": {"revenue": 2.5, "quarter": 2.0, "earnings": 2.5, "margin": 2.0, "profit": 2.2, "cost": 1.8},
        "SUPPORT": {"ticket": 2.5, "help": 2.0, "assist": 1.8, "agent": 1.5, "customer": 2.0, "issue": 1.8},
    }

    def __init__(self, model_name: str = "distilbert_sentiment_classifier") -> None:
        self.model_name = model_name

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Compute softmax probabilities with numerical stability."""
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)

    def classify(self, text: str, candidate_labels: Optional[List[str]] = None) -> ClassificationResult:
        """
        Classify text against provided candidate labels or default sentiment classes.
        """
        start_time = time.perf_counter()
        labels = candidate_labels if candidate_labels else self.DEFAULT_CLASSES
        words = text.lower().strip().split()

        raw_scores = []
        for label in labels:
            label_upper = label.upper()
            weights = self.LEXICON_WEIGHTS.get(label_upper, {})
            score = 0.5  # Base prior
            for word in words:
                if word in weights:
                    score += weights[word]
            raw_scores.append(score)

        logits = np.array(raw_scores, dtype=np.float32)
        probs = self._softmax(logits)

        predictions = [
            ClassificationPrediction(label=lbl, score=round(float(p), 4))
            for lbl, p in zip(labels, probs)
        ]

        # Sort descending by probability
        predictions = sorted(predictions, key=lambda p: p.score, reverse=True)
        top = predictions[0]

        latency = (time.perf_counter() - start_time) * 1000.0

        return ClassificationResult(
            source_text=text,
            top_label=top.label,
            top_score=top.score,
            probabilities=predictions,
            inference_latency_ms=round(latency, 2),
        )
