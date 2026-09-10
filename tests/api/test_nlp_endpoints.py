"""
API Integration tests for NLP REST Endpoints (/api/v1/nlp/*).
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_nlp_models(client: AsyncClient, admin_headers: dict):
    response = await client.get("/api/v1/nlp/models", headers=admin_headers)
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 4
    assert "PERSON" in data["entity_types"]


@pytest.mark.asyncio
async def test_embed_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "texts": ["OmniForge intelligence platform", "Real-time multimodal artificial intelligence"],
        "dimension": 384,
    }
    response = await client.post("/api/v1/nlp/embed", json=payload, headers=engineer_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_embeddings"] == 2
    assert data["dimension"] == 384
    assert len(data["embeddings"]) == 2
    assert len(data["embeddings"][0]["vector"]) == 384


@pytest.mark.asyncio
async def test_ner_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "text": "Satya Nadella spoke at Microsoft headquarters in Seattle on October 10.",
        "min_confidence": 0.50,
    }
    response = await client.post("/api/v1/nlp/ner", json=payload, headers=engineer_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_entities"] > 0
    labels = [e["label"] for e in data["entities"]]
    assert "ORG" in labels
    assert "GPE" in labels


@pytest.mark.asyncio
async def test_classify_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "text": "The platform architecture is exceptionally well-designed and reliable.",
        "candidate_labels": ["POSITIVE", "NEUTRAL", "NEGATIVE"],
    }
    response = await client.post("/api/v1/nlp/classify", json=payload, headers=engineer_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["top_label"] == "POSITIVE"
    assert data["top_score"] > 0.5
    assert len(data["probabilities"]) == 3


@pytest.mark.asyncio
async def test_similarity_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "query": "Machine learning model optimization",
        "documents": [
            "Training neural networks on GPU clusters",
            "Cooking pasta with tomato sauce recipe",
            "Deep learning inference acceleration",
        ],
        "top_k": 2,
    }
    response = await client.post("/api/v1/nlp/similarity", json=payload, headers=engineer_headers)
    assert response.status_code == 200
    data = response.json()
    assert data["total_documents"] == 3
    assert len(data["top_k_matches"]) == 2
    assert data["top_k_matches"][0]["similarity_score"] >= data["top_k_matches"][1]["similarity_score"]
