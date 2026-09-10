"""
Unit tests for text classification and sentiment analysis.
"""

import pytest

from nlp.classification import TextClassifier


def test_text_classifier_sentiment():
    classifier = TextClassifier()

    pos_text = "The platform delivers excellent performance and great user experience with robust APIs."
    res_pos = classifier.classify(pos_text)

    assert res_pos.top_label == "POSITIVE"
    assert res_pos.top_score > 0.5

    # Check that probability distribution sums to ~1.0
    total_prob = sum(p.score for p in res_pos.probabilities)
    assert total_prob == pytest.approx(1.0, rel=1e-2)

    neg_text = "Severe bug in database connection causes terrible crash and fatal error."
    res_neg = classifier.classify(neg_text)
    assert res_neg.top_label == "NEGATIVE"
    assert res_neg.top_score > 0.5


def test_text_classifier_custom_candidate_labels():
    classifier = TextClassifier()
    text = "Our fiscal quarterly revenue surged 25% with strong profit margins."
    candidate_labels = ["FINANCE", "TECHNOLOGY", "SUPPORT"]

    res = classifier.classify(text, candidate_labels=candidate_labels)
    assert res.top_label == "FINANCE"
    assert len(res.probabilities) == 3
