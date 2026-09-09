"""
Unit tests for Declarative Tool Calling (@tool), ToolRegistry, and execution safety.
"""

import pytest
from agents.tools import FunctionTool, ToolRegistry, tool


def test_tool_decorator_introspection():
    @tool(name="custom_math_tool", description="Add two integers with optional multiplier", category="math")
    def custom_math(a: int, b: int, multiplier: int = 1) -> int:
        return (a + b) * multiplier

    registry = ToolRegistry.get_instance()
    tool_obj = registry.get("custom_math_tool")

    assert tool_obj is not None
    assert tool_obj.definition.name == "custom_math_tool"
    assert tool_obj.definition.category == "math"
    assert len(tool_obj.definition.parameters) == 3

    param_names = [p.name for p in tool_obj.definition.parameters]
    assert "a" in param_names
    assert "b" in param_names
    assert "multiplier" in param_names

    # Check JSON Schema structure
    schema = tool_obj.definition.to_json_schema()
    assert schema["name"] == "custom_math_tool"
    assert "a" in schema["parameters"]["properties"]
    assert "multiplier" in schema["parameters"]["properties"]
    assert "a" in schema["parameters"]["required"]
    assert "multiplier" not in schema["parameters"]["required"]


def test_tool_execution_success_and_latency():
    registry = ToolRegistry.get_instance()
    res = registry.execute("ml_predict", {"model_type": "classification", "features": [{"monthly_charges": 80}]})

    assert res.success is True
    assert res.error is None
    assert res.output["model_type"] == "classification"
    assert res.output["prediction"] == [1]
    assert res.latency_ms >= 0.0


def test_tool_execution_unregistered_tool():
    registry = ToolRegistry.get_instance()
    res = registry.execute("non_existent_tool_123", {"param": 1})

    assert res.success is False
    assert "not registered" in res.error


def test_tool_execution_runtime_exception_handling():
    @tool(name="failing_tool", description="A tool that raises an exception")
    def failing_tool(x: int) -> int:
        if x == 0:
            raise ValueError("Division by zero error simulated")
        return 10 // x

    registry = ToolRegistry.get_instance()
    res = registry.execute("failing_tool", {"x": 0})

    assert res.success is False
    assert "Division by zero error simulated" in res.error
    assert res.latency_ms >= 0.0
