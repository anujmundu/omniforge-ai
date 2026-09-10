"""
Pydantic v2 schemas for Multi-Agent Orchestration REST APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class AgentStepSchema(BaseModel):
    step_index: int
    thought: str
    action: Optional[Dict[str, Any]] = None
    observation: Optional[Dict[str, Any]] = None
    timestamp: float


class AgentPlanStepSchema(BaseModel):
    step_id: int
    description: str
    assigned_agent: str
    dependencies: List[int] = Field(default_factory=list)
    status: str
    output: Optional[Any] = None


class AgentPlanSchema(BaseModel):
    plan_id: str
    user_intent: str
    steps: List[AgentPlanStepSchema]


class ToolExecutionResultSchema(BaseModel):
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    success: bool
    error: Optional[str] = None
    latency_ms: float


class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question or complex multi-modal prompt")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional payload / domain parameters")
    session_id: Optional[str] = Field(default=None, description="Conversational session identifier")


class AgentChatResponse(BaseModel):
    query: str
    final_answer: str
    plan: Optional[AgentPlanSchema] = None
    steps: List[AgentStepSchema] = Field(default_factory=list)
    tool_calls: List[ToolExecutionResultSchema] = Field(default_factory=list)
    latency_ms: float
    status: str


class DecomposePlanRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Complex prompt to decompose into DAG plan")


class DecomposePlanResponse(BaseModel):
    plan: AgentPlanSchema


class ToolParameterSchema(BaseModel):
    name: str
    type: str
    description: str
    required: bool
    default: Optional[Any] = None


class ToolDefinitionSchema(BaseModel):
    name: str
    description: str
    category: str
    parameters: List[ToolParameterSchema]
    json_schema: Dict[str, Any]


class ListToolsResponse(BaseModel):
    total_tools: int
    tools: List[ToolDefinitionSchema]


class ExecuteToolRequest(BaseModel):
    tool_name: str = Field(..., min_length=1)
    arguments: Dict[str, Any] = Field(default_factory=dict)


class ExecuteToolResponse(BaseModel):
    result: ToolExecutionResultSchema
