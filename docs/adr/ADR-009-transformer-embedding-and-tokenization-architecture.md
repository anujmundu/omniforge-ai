# ADR 009: Transformer Embedding and Tokenization Architecture

## Status
Accepted

## Date
2026-09-07

## Context
OmniForge requires high-throughput semantic text representation for document classification, retrieval-augmented generation (RAG), semantic clustering, and cross-lingual search.
Key requirements include:
1. Standardized 384 / 768 / 1536-dimensional dense vector representation.
2. $L_2$ unit normalization ensuring cosine similarity can be calculated via fast dot products.
3. Deterministic batching with sub-millisecond vector math.
4. Pluggable transformer backends (Sentence-Transformers, Hugging Face, OpenAI, and lightweight embedded tokenizers).

## Decision
We implement a decoupled embedding architecture in `nlp/`:
1. **Contract (`nlp/base.py`)**: `BaseEmbeddingModel` abstract class returning typed `TextEmbedding` and `BatchEmbeddingResult` structures.
2. **Embedding Engine (`nlp/embeddings.py`)**: Vectorizer implementing sub-word / n-gram hashed semantic hashing with exact unit-norm scaling and batch matrix dot-product cosine similarity calculators.

## Consequences
- Single unified contract for both local edge models and cloud API vectorizers.
- Zero extra network roundtrips for cosine distance calculations.
- Clean vector compatibility with downstream Vector Databases (ChromaDB, PostgreSQL pgvector).
