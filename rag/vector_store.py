"""
In-Memory & Persistent Vector Store Collection Manager for Enterprise RAG.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from nlp.embeddings import TransformerEmbeddingEngine
from rag.base import BaseVectorStore, DocumentChunk, RetrievalResult


class InMemoryVectorStore(BaseVectorStore):
    """
    High-performance Vector Store for dense embeddings.
    Supports named collections, cosine distance indexing, and metadata filtering.
    """

    def __init__(self, embedder: Optional[TransformerEmbeddingEngine] = None) -> None:
        self.embedder = embedder or TransformerEmbeddingEngine(dimension=384)
        # collection_name -> List[DocumentChunk]
        self._collections: Dict[str, List[DocumentChunk]] = {}
        # collection_name -> numpy matrix of vectors
        self._matrices: Dict[str, np.ndarray] = {}

    def add_chunks(self, collection_name: str, chunks: List[DocumentChunk]) -> int:
        """
        Index chunks into the specified collection, computing embeddings if not present.
        """
        if collection_name not in self._collections:
            self._collections[collection_name] = []
            self._matrices[collection_name] = np.empty((0, self.embedder.dimension), dtype=np.float32)

        if not chunks:
            return 0

        # Generate embeddings for chunks missing vectors
        unembedded_texts = [c.text for c in chunks if c.vector is None]
        if unembedded_texts:
            batch_embs = self.embedder.embed_batch(unembedded_texts)
            emb_idx = 0
            for c in chunks:
                if c.vector is None:
                    c.vector = batch_embs.embeddings[emb_idx].vector
                    emb_idx += 1

        # Append to collection
        self._collections[collection_name].extend(chunks)
        new_vecs = np.array([c.vector for c in chunks], dtype=np.float32)

        if self._matrices[collection_name].shape[0] == 0:
            self._matrices[collection_name] = new_vecs
        else:
            self._matrices[collection_name] = np.vstack([self._matrices[collection_name], new_vecs])

        return len(chunks)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievalResult]:
        """
        Search collection by dense cosine similarity.
        """
        if collection_name not in self._collections or len(self._collections[collection_name]) == 0:
            return []

        chunks = self._collections[collection_name]
        matrix = self._matrices[collection_name]
        q_vec = np.array(query_vector, dtype=np.float32)

        # Dot product (since both matrix rows and q_vec are L2 unit normalized)
        scores = np.dot(matrix, q_vec)
        scores = np.clip(scores, -1.0, 1.0)

        # Apply metadata filters if provided
        valid_indices = []
        for idx, chunk in enumerate(chunks):
            if metadata_filter:
                match = True
                for k, v in metadata_filter.items():
                    if chunk.metadata.get(k) != v and getattr(chunk, k, None) != v:
                        match = False
                        break
                if match:
                    valid_indices.append(idx)
            else:
                valid_indices.append(idx)

        if not valid_indices:
            return []

        # Sort filtered candidates descending by similarity score
        valid_scores = [(idx, float(scores[idx])) for idx in valid_indices]
        valid_scores.sort(key=lambda item: item[1], reverse=True)

        top_matches = valid_scores[:top_k]
        results: List[RetrievalResult] = []

        for rank, (idx, score) in enumerate(top_matches, 1):
            results.append(
                RetrievalResult(
                    chunk=chunks[idx],
                    similarity_score=round(score, 4),
                    rank=rank,
                )
            )

        return results

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all active collections, chunk counts, and dimension info.
        """
        collections_info = []
        for name, chunk_list in self._collections.items():
            doc_ids = len(set(c.doc_id for c in chunk_list))
            collections_info.append(
                {
                    "collection_name": name,
                    "total_chunks": len(chunk_list),
                    "unique_documents": doc_ids,
                    "vector_dimension": self.embedder.dimension,
                }
            )
        return collections_info

    def clear(self, collection_name: Optional[str] = None) -> None:
        """Clear a specific collection or all collections."""
        if collection_name:
            self._collections.pop(collection_name, None)
            self._matrices.pop(collection_name, None)
        else:
            self._collections.clear()
            self._matrices.clear()
