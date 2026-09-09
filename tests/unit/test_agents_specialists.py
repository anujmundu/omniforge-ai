"""
Unit tests for Domain Specialist Agents (ML, Vision, NLP, RAG).
"""

import pytest
from agents.specialists import (
    EnterpriseRAGAgent,
    MLAnalyticsAgent,
    NLPProcessingAgent,
    VisionAnalyticsAgent,
)


def test_ml_analytics_agent():
    agent = MLAnalyticsAgent()
    res = agent.run("Predict customer churn probability")

    assert res.status == "COMPLETED"
    assert len(res.steps) == 1
    assert len(res.tool_calls) == 1
    assert "ML Classification" in res.final_answer
    assert res.latency_ms > 0.0


def test_vision_analytics_agent():
    agent = VisionAnalyticsAgent()
    res_detect = agent.run("Detect objects in image frame")
    assert "Object Detection complete" in res_detect.final_answer

    res_ocr = agent.run("Extract invoice line items with OCR")
    assert "Spatial OCR Extraction" in res_ocr.final_answer


def test_nlp_processing_agent():
    agent = NLPProcessingAgent()
    res = agent.run("Extract named entities from: Satya Nadella visited Microsoft headquarters in London")

    assert res.status == "COMPLETED"
    assert "NLP Entity Extraction" in res.final_answer
    assert "Microsoft" in res.final_answer or "Satya Nadella" in res.final_answer


def test_enterprise_rag_agent():
    agent = EnterpriseRAGAgent()
    res = agent.run("Search knowledge base for OmniForge architecture overview")

    assert res.status == "COMPLETED"
    assert "RAG Knowledge Base Search" in res.final_answer
    assert len(res.tool_calls) == 1
