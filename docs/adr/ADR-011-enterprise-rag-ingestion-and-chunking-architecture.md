# ADR 011: Enterprise RAG Ingestion and Chunking Architecture

## Status
Accepted

## Date
2026-09-08

## Context
Enterprise knowledge bases contain heterogeneous unstructured documents (Markdown reports, technical specifications, policy PDFs, JSON records) requiring semantic decomposition. Naive fixed-character chunking breaks sentence context and cuts critical tabular or technical data across arbitrary boundaries.

## Decision
We implement a robust, structure-aware ingestion and chunking engine in `rag/`:
1. **Multi-Format Parsing (`rag/parser.py`)**: Unified `Document` abstraction retaining source path, document title, author, and custom metadata.
2. **Recursive Semantic Chunker (`rag/chunker.py`)**:
   - Respects natural document hierarchy (`\n\n` paragraphs &rarr; `\n` linebreaks &rarr; `.` sentences &rarr; ` ` words).
   - Configurable chunk token capacity (default 500 characters) and sliding token overlap (default 100 characters).
   - Generates deterministic SHA-256 chunk identifiers (`{doc_id}_{chunk_index}`).

## Consequences
- Preserves semantic context within each indexed chunk.
- Guarantees zero orphaned fragments and provides predictable token windows for embedding models and LLMs.
