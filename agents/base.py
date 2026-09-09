"""
Base Domain Models, Schemas, and Abstract Interfaces for OmniForge Multi-Agent Mesh.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field


class AgentRole(str, Enum):
    SUPERVISOR = "SUPERVISOR"
    ML_SPECIALIST = "ML_SPECIALIST"
    VISION_SPECIALIST = "VISION_SPECIALIST"
    NLP_SPECIALIST = "NLP_SPECIALIST"
    RAG_SPECIALIST = "RAG_SPECIALIST"
    TOOL_CALLER = "TOOL_CALLER"


class AgentMessage(BaseModel):
    """Conversational turn message."""
    role: str = Field(..., description="user | assistant | system | tool")
    content: str = Field(..., description="Message text content")
    name: Optional[str] = Field(default=None, description="Agent or tool name identifier")
    timestamp: float = Field(default_factory=time.time)


class ToolParameter(BaseModel):
    """Specification of a single tool argument."""
    name: str
    type: str = Field(default="string", description="string | integer | number | boolean | object | array")
    description: str
    required: bool = True
    default: Optional[Any] = None


class ToolDefinition(BaseModel):
    """Introspected metadata and JSON Schema definition of a registered tool."""
    name: str
    description: str
    parameters: List[ToolParameter] = Field(default_factory=list)
    category: str = Field(default="general", description="ml | vision | nlp | rag | sql | general")

    def to_json_schema(self) -> Dict[str, Any]:
        """Convert tool specification to standard JSON Schema dictionary."""
        properties: Dict[str, Any] = {}
        required_fields: List[str] = []

        for p in self.parameters:
            properties[p.name] = {
                "type": p.type,
                "description": p.description,
            }
            if p.default is not None:
                properties[p.name]["default"] = p.default
            if p.required:
                required_fields.append(p.name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            },
        }


class ToolExecutionResult(BaseModel):
    """Result of an executed tool invocation."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)
    output: Any = None
    success: bool = True
    error: Optional[str] = None
    latency_ms: float = 0.0


class AgentAction(BaseModel):
    """Action selected by an agent in a ReAct loop."""
    tool_name: str
    arguments: Dict[str, Any] = Field(default_factory=dict)


class AgentObservation(BaseModel):
    """Observation observed from tool or specialist execution."""
    output: Any = None
    error: Optional[str] = None
    success: bool = True


class AgentStep(BaseModel):
    """A single ReAct reasoning cycle step."""
    step_index: int = Field(..., ge=0)
    thought: str = Field(..., description="Agent internal chain-of-thought reasoning")
    action: Optional[AgentAction] = None
    observation: Optional[AgentObservation] = None
    timestamp: float = Field(default_factory=time.time)


class AgentPlanStep(BaseModel):
    """A decomposed sub-task in an overall multi-agent execution plan."""
    step_id: int
    description: str
    assigned_agent: str = Field(default="GENERAL")
    dependencies: List[int] = Field(default_factory=list)
    status: str = Field(default="PENDING", description="PENDING | IN_PROGRESS | COMPLETED | FAILED")
    output: Optional[Any] = None


class AgentPlan(BaseModel):
    """Structured DAG decomposition of a multi-modal user request."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:10]}")
    user_intent: str
    steps: List[AgentPlanStep] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class AgentResponse(BaseModel):
    """Complete multi-agent execution response."""
    query: str
    final_answer: str
    plan: Optional[AgentPlan] = None
    steps: List[AgentStep] = Field(default_factory=list)
    tool_calls: List[ToolExecutionResult] = Field(default_factory=list)
    latency_ms: float = 0.0
    status: str = Field(default="COMPLETED", description="COMPLETED | FAILED | MAX_STEPS_REACHED")


# ==============================================================================
# Abstract Agent & Tool Interfaces
# ==============================================================================

class BaseTool(ABC):
    """Abstract interface for executable tools."""

    @property
    @abstractmethod
    def definition(self) -> ToolDefinition:
        """Return the introspected definition of this tool."""
        pass

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolExecutionResult:
        """Execute the tool logic with validated arguments."""
        pass


class BaseAgent(ABC):
    """Abstract interface for autonomous agents and specialists."""

    def __init__(self, name: str, role: AgentRole, description: str) -> None:
        self.name = name
        self.role = role
        self.description = description

    @abstractmethod
    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Execute agent workflow on input query."""
        pass
