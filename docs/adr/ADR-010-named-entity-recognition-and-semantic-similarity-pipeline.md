# ADR 010: Named Entity Recognition and Semantic Similarity Pipeline

## Status
Accepted

## Date
2026-09-07

## Context
Industrial unstructured text processing requires entity extraction with character span localization (for highlighting in UI and downstream extraction pipelines) and semantic distance metrics across heterogeneous documents.

## Decision
1. **Named Entity Recognition (`nlp/ner.py`)**:
   - Extraction of entity categories (`PERSON`, `ORG`, `GPE`, `MONEY`, `DATE`, `TECH_STACK`, `PRODUCT`).
   - Character span indexing: guarantee that `source_text[start:end] == span_text`.
   - Confidence scoring per entity.
2. **Text Classification & Sentiment (`nlp/classification.py`)**:
   - Normalized probability distributions across class labels (sum to 1.0 via Softmax).
   - High-throughput multi-class routing.
3. **Cross-Document Semantic Similarity (`nlp/similarity.py`)**:
   - $N \times N$ symmetric cosine similarity matrix calculation.
   - Top-K nearest document retrieval with relevance thresholding.

## Consequences
- Enables downstream agents and RAG pipelines to perform precision entity filtering and semantic ranking.
- Sub-5ms latency per document for classification and entity extraction.
