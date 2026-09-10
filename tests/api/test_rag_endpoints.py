"""
API Integration tests for Enterprise RAG Endpoints (/api/v1/rag/*).
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_rag_index_and_collections_flow(client: AsyncClient, engineer_headers: dict):
    # 1. Index documents
    index_payload = {
        "collection_name": "api_test_kb",
        "documents": [
            {
                "title": "FastAPI Gateway",
                "content": "FastAPI provides asynchronous request handling with Pydantic schema validation.",
                "source_type": "text",
                "metadata": {"layer": "api"},
            },
            {
                "title": "PostgreSQL Storage",
                "content": "PostgreSQL is used as the relational metadata storage layer.",
                "source_type": "text",
                "metadata": {"layer": "database"},
            },
        ],
    }
    index_res = await client.post("/api/v1/rag/documents/index", json=index_payload, headers=engineer_headers)
    assert index_res.status_code == 200
    index_data = index_res.json()
    assert index_data["collection_name"] == "api_test_kb"
    assert index_data["total_documents"] == 2
    assert index_data["total_chunks_created"] >= 2

    # 2. List collections
    colls_res = await client.get("/api/v1/rag/collections", headers=engineer_headers)
    assert colls_res.status_code == 200
    colls_data = colls_res.json()
    names = [c["collection_name"] for c in colls_data["collections"]]
    assert "api_test_kb" in names


@pytest.mark.asyncio
async def test_rag_retrieve_endpoint(client: AsyncClient, engineer_headers: dict):
    # Retrieve relevant chunks
    retrieve_payload = {
        "query": "asynchronous request handling and validation",
        "collection_name": "api_test_kb",
        "top_k": 2,
        "rerank": True,
    }
    res = await client.post("/api/v1/rag/retrieve", json=retrieve_payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["query"] == "asynchronous request handling and validation"
    assert len(data["results"]) >= 1
    assert data["results"][0]["chunk"]["title"] == "FastAPI Gateway"
    assert data["results"][0]["rerank_score"] is not None


@pytest.mark.asyncio
async def test_rag_query_endpoint(client: AsyncClient, engineer_headers: dict):
    # Grounded query with citations
    query_payload = {
        "query": "How does FastAPI handle API requests?",
        "collection_name": "api_test_kb",
        "top_k": 2,
        "rerank": True,
    }
    res = await client.post("/api/v1/rag/query", json=query_payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert len(data["citations"]) >= 1
    assert data["citations"][0]["doc_title"] == "FastAPI Gateway"
    assert "[1]" in data["answer"]


@pytest.mark.asyncio
async def test_rag_evaluate_endpoint(client: AsyncClient, engineer_headers: dict):
    # Evaluate RAG answer
    eval_payload = {
        "query": "What is FastAPI used for?",
        "generated_answer": "FastAPI provides asynchronous request handling [1].",
        "retrieved_chunk_texts": ["FastAPI provides asynchronous request handling with Pydantic schema validation."],
        "ground_truth_answer": "FastAPI handles asynchronous web requests.",
    }
    res = await client.post("/api/v1/rag/evaluate", json=eval_payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["faithfulness_score"] >= 0.70
    assert data["answer_relevance_score"] >= 0.70
    assert data["context_precision_score"] >= 0.80
    assert data["overall_rag_score"] >= 0.70
