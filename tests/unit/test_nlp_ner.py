"""
Unit tests for Named Entity Recognition (NER).
"""

import pytest
from nlp.ner import NamedEntityRecognizer


def test_ner_exact_character_span_offsets():
    ner = NamedEntityRecognizer()
    text = "Sundar Pichai visited Google headquarters in San Francisco on September 15 with $50,000 in funding for Python systems."

    result = ner.extract_entities(text, min_confidence=0.5)

    assert result.total_entities > 0
    assert len(result.entities) > 0

    # Verify character indexing integrity: text[start:end] == span.text
    for span in result.entities:
        sliced_text = text[span.start_char:span.end_char]
        assert sliced_text == span.text, f"Offset mismatch: expected '{span.text}', got '{sliced_text}'"
        assert span.confidence >= 0.5


def test_ner_entity_types():
    ner = NamedEntityRecognizer()
    text = "Microsoft launched a new PostgreSQL cloud service in London on March 20."

    result = ner.extract_entities(text)
    labels = {e.label for e in result.entities}

    assert "ORG" in labels  # Microsoft
    assert "GPE" in labels  # London
    assert "DATE" in labels # March 20
    assert "TECH_STACK" in labels # PostgreSQL
