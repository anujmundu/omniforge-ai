"""
Cross-Document Semantic Similarity Engine and Top-K Nearest Document Retrieval.
"""

from __future__ import annotations

import time
from typing import List, Optional

import numpy as np

from nlp.base import BaseSimilarityEngine, RankedDocument, SimilarityMatrixResult
from nlp.embeddings import TransformerEmbeddingEngine


class SemanticSimilarityEngine(BaseSimilarityEngine):
    """
    High-throughput semantic similarity engine.
    Computes dense cosine similarity matrices and performs top-K nearest neighbor document search.
    """

    def __init__(self, embedding_engine: Optional[TransformerEmbeddingEngine] = None) -> None:
        self.embedder = embedding_engine or TransformerEmbeddingEngine()

    def compute_similarity_matrix(self, texts: List[str]) -> SimilarityMatrixResult:
        """
        Compute symmetric N x N pairwise cosine similarity matrix.
        """
        start_time = time.perf_counter()
        if not texts:
            return SimilarityMatrixResult(documents=[], similarity_matrix=[], inference_latency_ms=0.0)

        batch_result = self.embedder.embed_batch(texts)
        vectors = np.array([emb.vector for emb in batch_result.embeddings], dtype=np.float32)

        # Pairwise dot product (cosine similarity since vectors are unit-normalized)
        sim_matrix = np.dot(vectors, vectors.T)

        # Enforce exact floating point bounds [-1.0, 1.0] and diagonal 1.0
        np.clip(sim_matrix, -1.0, 1.0, out=sim_matrix)
        np.fill_diagonal(sim_matrix, 1.0)

        matrix_list = [[round(float(val), 4) for val in row] for row in sim_matrix]

        latency = (time.perf_counter() - start_time) * 1000.0

        return SimilarityMatrixResult(
            documents=texts,
            similarity_matrix=matrix_list,
            inference_latency_ms=round(latency, 2),
        )

    def search_top_k(self, query: str, documents: List[str], top_k: int = 5) -> SimilarityMatrixResult:
        """
        Rank candidate documents by cosine similarity to the query string.
        """
        start_time = time.perf_counter()
        if not documents:
            return SimilarityMatrixResult(
                query_text=query, documents=[], similarity_matrix=[], top_k_matches=[], inference_latency_ms=0.0
            )

        query_emb = self.embedder.embed_text(query)
        doc_batch = self.embedder.embed_batch(documents)

        q_vec = np.array(query_emb.vector, dtype=np.float32)
        doc_vecs = np.array([emb.vector for emb in doc_batch.embeddings], dtype=np.float32)

        scores = np.dot(doc_vecs, q_vec)
        scores = np.clip(scores, -1.0, 1.0)

        # Pair documents with scores
        ranked = [
            RankedDocument(document_index=idx, text=documents[idx], similarity_score=round(float(scores[idx]), 4))
            for idx in range(len(documents))
        ]

        # Sort descending by score
        ranked.sort(key=lambda r: r.similarity_score, reverse=True)
        top_matches = ranked[:top_k]

        latency = (time.perf_counter() - start_time) * 1000.0

        return SimilarityMatrixResult(
            query_text=query,
            documents=documents,
            similarity_matrix=[[round(float(s), 4) for s in scores]],
            top_k_matches=top_matches,
            inference_latency_ms=round(latency, 2),
        )
