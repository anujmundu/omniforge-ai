"""
Unit tests for SupervisorAgent (Plan decomposition, ReAct loop, Multi-Agent Synthesis).
"""

import pytest
from agents.orchestrator import SupervisorAgent


@pytest.fixture
def supervisor():
    return SupervisorAgent()


def test_supervisor_decompose_plan(supervisor):
    complex_prompt = (
        "Extract total amounts from invoice image using OCR, "
        "extract named entity organizations with NLP, "
        "search documentation guide with RAG, "
        "and predict spending anomaly with ML."
    )
    plan = supervisor.decompose_plan(complex_prompt)

    assert len(plan.steps) >= 4
    agents_assigned = [s.assigned_agent for s in plan.steps]
    assert "VisionAnalyticsAgent" in agents_assigned
    assert "NLPProcessingAgent" in agents_assigned
    assert "EnterpriseRAGAgent" in agents_assigned
    assert "MLAnalyticsAgent" in agents_assigned


def test_supervisor_full_react_execution(supervisor):
    query = "Search knowledge base for OmniForge architecture and predict customer churn."
    res = supervisor.run(query)

    assert res.status == "COMPLETED"
    assert res.plan is not None
    assert len(res.plan.steps) >= 2
    assert len(res.steps) >= 2
    assert len(res.tool_calls) >= 2
    assert "Multi-Agent Orchestrator successfully executed" in res.final_answer
    assert res.latency_ms > 0.0
