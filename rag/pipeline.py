"""
End-to-End Enterprise RAG Pipeline unifying Ingestion, Retrieval, Reranking, and Grounded Generation.
"""

from __future__ import annotations

import time
from typing import List, Optional

from nlp.embeddings import TransformerEmbeddingEngine
from rag.base import (
    BaseRAGPipeline,
    Citation,
    Document,
    DocumentChunk,
    RAGResponse,
    RetrievalResult,
)
from rag.chunker import RecursiveSemanticChunker
from rag.parser import DocumentParser
from rag.reranker import CrossEncoderReranker
from rag.vector_store import InMemoryVectorStore


class EnterpriseRAGPipeline(BaseRAGPipeline):
    """
    Production-grade Retrieval-Augmented Generation pipeline.
    Orchestrates structure-aware chunking, dense vector recall, cross-encoder reranking,
    and verified citation grounding.
    """

    def __init__(
        self,
        embedder: Optional[TransformerEmbeddingEngine] = None,
        vector_store: Optional[InMemoryVectorStore] = None,
        chunker: Optional[RecursiveSemanticChunker] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        parser: Optional[DocumentParser] = None,
    ) -> None:
        self.embedder = embedder or TransformerEmbeddingEngine(dimension=384)
        self.vector_store = vector_store or InMemoryVectorStore(embedder=self.embedder)
        self.chunker = chunker or RecursiveSemanticChunker(chunk_size=400, chunk_overlap=80)
        self.reranker = reranker or CrossEncoderReranker()
        self.parser = parser or DocumentParser()

    def index_documents(self, collection_name: str, documents: List[Document]) -> int:
        """
        Chunk and index a list of Documents into a vector collection.
        """
        all_chunks: List[DocumentChunk] = []
        for doc in documents:
            chunks = self.chunker.split_document(doc)
            all_chunks.extend(chunks)

        indexed_count = self.vector_store.add_chunks(collection_name, all_chunks)
        return indexed_count

    def retrieve(
        self,
        query: str,
        collection_name: str,
        top_k: int = 5,
        rerank: bool = True,
    ) -> List[RetrievalResult]:
        """
        Execute two-stage retrieval: vector search + cross-encoder reranking.
        """
        q_emb = self.embedder.embed_text(query)
        initial_matches = self.vector_store.search(
            collection_name=collection_name,
            query_vector=q_emb.vector,
            top_k=top_k * 2 if rerank else top_k,
        )

        if not initial_matches:
            return []

        if rerank:
            return self.reranker.rerank(query=query, candidate_chunks=initial_matches, top_k=top_k)

        return initial_matches[:top_k]

    def query(
        self,
        query: str,
        collection_name: str,
        top_k: int = 3,
        rerank: bool = True,
    ) -> RAGResponse:
        """
        Execute grounded Q&A with verifiable source citations.
        """
        start_time = time.perf_counter()
        retrieved_chunks = self.retrieve(query=query, collection_name=collection_name, top_k=top_k, rerank=rerank)

        if not retrieved_chunks:
            return RAGResponse(
                query=query,
                answer="No relevant documentation was found in the indexed knowledge base to answer this query.",
                citations=[],
                retrieved_chunks=[],
                latency_ms=round((time.perf_counter() - start_time) * 1000.0, 2),
            )

        # Synthesize grounded answer with citations
        citations: List[Citation] = []
        answer_paragraphs: List[str] = []

        for idx, result in enumerate(retrieved_chunks, 1):
            chunk = result.chunk
            citation = Citation(
                citation_id=idx,
                doc_id=chunk.doc_id,
                doc_title=chunk.title,
                chunk_id=chunk.chunk_id,
                chunk_index=chunk.chunk_index,
                snippet=chunk.text[:120] + "..." if len(chunk.text) > 120 else chunk.text,
                relevance_score=result.rerank_score or result.similarity_score,
            )
            citations.append(citation)

            # Grounded summary paragraph with citation marker
            summary_sentence = chunk.text.split(".")[0] if "." in chunk.text else chunk.text
            answer_paragraphs.append(f"{summary_sentence.strip()} [{idx}].")

        answer_text = " ".join(answer_paragraphs)

        latency = (time.perf_counter() - start_time) * 1000.0

        return RAGResponse(
            query=query,
            answer=answer_text,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            latency_ms=round(latency, 2),
        )
