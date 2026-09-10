"""
Enterprise RAG API Router for Ingestion, Hybrid Retrieval, Cross-Encoder Reranking,
Grounded Q&A Generation, and Automated Evaluation.
"""

from __future__ import annotations

import time
from typing import Any, List

from fastapi import APIRouter, Depends

from apps.api.core.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.schemas.rag import (
    CitationSchema,
    CollectionInfoSchema,
    DocumentChunkSchema,
    IndexDocumentsRequest,
    IndexDocumentsResponse,
    ListCollectionsResponse,
    RAGEvaluateRequest,
    RAGEvaluateResponse,
    RAGQueryRequest,
    RAGQueryResponse,
    RetrievalItemSchema,
    RetrieveRequest,
    RetrieveResponse,
)
from rag.base import Document, DocumentChunk, RetrievalResult
from rag.evaluator import RAGEvaluator
from rag.parser import DocumentParser
from rag.pipeline import EnterpriseRAGPipeline

router = APIRouter(prefix="/rag", tags=["Enterprise Retrieval-Augmented Generation (RAG)"])

# In-memory singleton pipeline for fast serving and collection management
_rag_pipeline = EnterpriseRAGPipeline()
_rag_evaluator = RAGEvaluator()
_doc_parser = DocumentParser()


@router.get("/collections", response_model=ListCollectionsResponse)
async def list_rag_collections(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    List all indexed vector collections, chunk counts, and dimensions.
    """
    colls = _rag_pipeline.vector_store.list_collections()
    items = [
        CollectionInfoSchema(
            collection_name=c["collection_name"],
            total_chunks=c["total_chunks"],
            unique_documents=c["unique_documents"],
            vector_dimension=c["vector_dimension"],
        )
        for c in colls
    ]
    return ListCollectionsResponse(
        collections=items,
        total_collections=len(items),
    )


@router.post("/documents/index", response_model=IndexDocumentsResponse)
async def index_documents(
    request: IndexDocumentsRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Parse, chunk, embed, and index a batch of documents into the specified collection.
    """
    start_time = time.perf_counter()
    parsed_docs: List[Document] = []

    for doc_in in request.documents:
        if doc_in.source_type == "markdown":
            doc = _doc_parser.parse_markdown(doc_in.content, title=doc_in.title, metadata=doc_in.metadata)
        elif doc_in.source_type == "json":
            doc = _doc_parser.parse_json(doc_in.content, title=doc_in.title, metadata=doc_in.metadata)
        else:
            doc = _doc_parser.parse_text(doc_in.content, title=doc_in.title, metadata=doc_in.metadata)

        if doc_in.doc_id:
            doc.doc_id = doc_in.doc_id
        parsed_docs.append(doc)

    indexed_chunks = _rag_pipeline.index_documents(
        collection_name=request.collection_name,
        documents=parsed_docs,
    )

    latency = round((time.perf_counter() - start_time) * 1000.0, 2)

    return IndexDocumentsResponse(
        collection_name=request.collection_name,
        total_documents=len(parsed_docs),
        total_chunks_created=indexed_chunks,
        dimension=_rag_pipeline.embedder.dimension,
        latency_ms=latency,
    )


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_context(
    request: RetrieveRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Retrieve top-K ranked chunks for a query using vector search and cross-encoder reranking.
    """
    start_time = time.perf_counter()
    results = _rag_pipeline.retrieve(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
        rerank=request.rerank,
    )

    items = [
        RetrievalItemSchema(
            chunk=DocumentChunkSchema(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                text=r.chunk.text,
                chunk_index=r.chunk.chunk_index,
                start_char=r.chunk.start_char,
                end_char=r.chunk.end_char,
                metadata=r.chunk.metadata,
            ),
            similarity_score=r.similarity_score,
            rerank_score=r.rerank_score,
            rank=r.rank,
        )
        for r in results
    ]

    latency = round((time.perf_counter() - start_time) * 1000.0, 2)

    return RetrieveResponse(
        query=request.query,
        collection_name=request.collection_name,
        total_retrieved=len(items),
        results=items,
        latency_ms=latency,
    )


@router.post("/query", response_model=RAGQueryResponse)
async def rag_query(
    request: RAGQueryRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Execute grounded RAG Q&A with structured citations.
    """
    response = _rag_pipeline.query(
        query=request.query,
        collection_name=request.collection_name,
        top_k=request.top_k,
        rerank=request.rerank,
    )

    citations = [
        CitationSchema(
            citation_id=c.citation_id,
            doc_id=c.doc_id,
            doc_title=c.doc_title,
            chunk_id=c.chunk_id,
            chunk_index=c.chunk_index,
            snippet=c.snippet,
            relevance_score=c.relevance_score,
        )
        for c in response.citations
    ]

    retrieved_chunks = [
        RetrievalItemSchema(
            chunk=DocumentChunkSchema(
                chunk_id=r.chunk.chunk_id,
                doc_id=r.chunk.doc_id,
                title=r.chunk.title,
                text=r.chunk.text,
                chunk_index=r.chunk.chunk_index,
                start_char=r.chunk.start_char,
                end_char=r.chunk.end_char,
                metadata=r.chunk.metadata,
            ),
            similarity_score=r.similarity_score,
            rerank_score=r.rerank_score,
            rank=r.rank,
        )
        for r in response.retrieved_chunks
    ]

    return RAGQueryResponse(
        query=response.query,
        answer=response.answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        model_name="OmniForge-HybridRAG-CrossEncoder",
        latency_ms=response.latency_ms,
    )


@router.post("/evaluate", response_model=RAGEvaluateResponse)
async def evaluate_rag(
    request: RAGEvaluateRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Quantitatively evaluate RAG response across faithfulness, answer relevance, and context precision.
    """
    retrieved_results = [
        RetrievalResult(
            chunk=DocumentChunk(
                chunk_id=f"eval_chunk_{i}",
                doc_id="eval_doc",
                title="Evaluation Context",
                text=text,
                chunk_index=i,
                start_char=0,
                end_char=len(text),
            ),
            similarity_score=0.9,
            rank=i + 1,
        )
        for i, text in enumerate(request.retrieved_chunk_texts)
    ]

    eval_result = _rag_evaluator.evaluate(
        query=request.query,
        generated_answer=request.generated_answer,
        retrieved_chunks=retrieved_results,
        ground_truth_answer=request.ground_truth_answer,
    )

    return RAGEvaluateResponse(
        query=eval_result.query,
        faithfulness_score=eval_result.faithfulness_score,
        answer_relevance_score=eval_result.answer_relevance_score,
        context_precision_score=eval_result.context_precision_score,
        overall_rag_score=eval_result.overall_rag_score,
        evaluation_details=eval_result.evaluation_details,
    )
