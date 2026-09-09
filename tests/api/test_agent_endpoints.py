"""
API Integration tests for Multi-Agent REST Endpoints (/api/v1/agents/*).
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_tools_endpoint(client: AsyncClient, engineer_headers: dict):
    res = await client.get("/api/v1/agents/tools", headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["total_tools"] >= 5
    tool_names = [t["name"] for t in data["tools"]]
    assert "ml_predict" in tool_names
    assert "vision_detect_objects" in tool_names
    assert "rag_search_knowledge_base" in tool_names


@pytest.mark.asyncio
async def test_execute_tool_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "tool_name": "ml_predict",
        "arguments": {
            "model_type": "classification",
            "features": [{"monthly_charges": 85.0}],
        },
    }
    res = await client.post("/api/v1/agents/execute-tool", json=payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["result"]["success"] is True
    assert data["result"]["tool_name"] == "ml_predict"
    assert data["result"]["output"]["prediction"] == [1]


@pytest.mark.asyncio
async def test_decompose_plan_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "query": "Extract invoice items with OCR, check customer churn prediction, and query knowledge base with RAG."
    }
    res = await client.post("/api/v1/agents/plan", json=payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert "plan" in data
    assert len(data["plan"]["steps"]) >= 3


@pytest.mark.asyncio
async def test_agent_chat_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "query": "Extract named entities with NLP, check database stats with SQL, and search policy documentation.",
        "context": {"department": "Data Science"},
    }
    res = await client.post("/api/v1/agents/chat", json=payload, headers=engineer_headers)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert "Multi-Agent Orchestrator successfully executed" in data["final_answer"]
    assert len(data["steps"]) >= 2
    assert len(data["tool_calls"]) >= 2
    assert data["latency_ms"] > 0.0
