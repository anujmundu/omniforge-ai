"""
Unit tests for AgentMemory (Short-term buffer, Scratchpad, Truncation).
"""

import pytest
from agents.base import AgentAction, AgentObservation, AgentStep
from agents.memory import AgentMemory


def test_agent_memory_buffer_truncation():
    memory = AgentMemory(max_history=3)
    memory.add_message(role="user", content="msg 1")
    memory.add_message(role="assistant", content="msg 2")
    memory.add_message(role="user", content="msg 3")
    memory.add_message(role="assistant", content="msg 4")

    assert len(memory.messages) == 3
    assert memory.messages[0].content == "msg 2"
    assert memory.messages[2].content == "msg 4"


def test_agent_memory_scratchpad_and_steps():
    memory = AgentMemory()
    memory.set_scratchpad("doc_id", "doc_123")
    memory.set_scratchpad("metric", 0.95)

    assert memory.get_scratchpad("doc_id") == "doc_123"
    assert memory.get_scratchpad("metric") == 0.95
    assert memory.get_scratchpad("missing_key", "default") == "default"

    step = AgentStep(
        step_index=1,
        thought="Test reasoning thought",
        action=AgentAction(tool_name="test_tool", arguments={"a": 1}),
        observation=AgentObservation(output={"result": "ok"}),
    )
    memory.add_step(step)
    assert len(memory.steps) == 1
    assert memory.steps[0].thought == "Test reasoning thought"

    memory.clear_scratchpad()
    assert len(memory.scratchpad) == 0
    assert len(memory.steps) == 0
