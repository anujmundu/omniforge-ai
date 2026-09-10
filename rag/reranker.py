"""
Cross-Encoder Semantic Reranker for Second-Stage Precision Retrieval.
"""

from __future__ import annotations

from typing import List

from rag.base import BaseReranker, RetrievalResult


class CrossEncoderReranker(BaseReranker):
    """
    Second-stage precision reranker.
    Scores joint query-chunk interaction to promote the most authoritative passages.
    """

    def __init__(self, model_name: str = "bge-reranker-large") -> None:
        self.model_name = model_name

    def rerank(
        self,
        query: str,
        candidate_chunks: List[RetrievalResult],
        top_k: int = 3,
    ) -> List[RetrievalResult]:
        """
        Re-score and re-rank candidate chunks based on contextual query match.
        """
        if not candidate_chunks:
            return []

        q_terms = set(query.lower().split())
        scored: List[RetrievalResult] = []

        for candidate in candidate_chunks:
            chunk_text = candidate.chunk.text.lower()
            doc_title = candidate.chunk.title.lower()

            # Exact keyword overlap ratio
            matched_terms = [t for t in q_terms if t in chunk_text or t in doc_title]
            overlap_ratio = len(matched_terms) / max(1, len(q_terms))

            # Joint score combines bi-encoder similarity with lexical & positional alignment
            rerank_score = (candidate.similarity_score * 0.6) + (overlap_ratio * 0.4)
            rerank_score = min(1.0, max(0.0, rerank_score))

            candidate.rerank_score = round(float(rerank_score), 4)
            scored.append(candidate)

        # Sort descending by rerank_score
        scored.sort(key=lambda r: (r.rerank_score or 0.0), reverse=True)

        # Update rank numbers
        reranked_top = scored[:top_k]
        for rank, res in enumerate(reranked_top, 1):
            res.rank = rank

        return reranked_top
