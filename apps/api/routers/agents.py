"""
Multi-Agent API Router for Intent Planning, Conversational ReAct Execution, and Tool Calling.
"""

from __future__ import annotations

import time
from typing import Any, List
from fastapi import APIRouter, Depends, status

from agents.memory import AgentMemory
from agents.orchestrator import SupervisorAgent
from agents.tools import ToolRegistry
from apps.api.core.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.schemas.agents import (
    AgentChatRequest,
    AgentChatResponse,
    AgentPlanSchema,
    AgentPlanStepSchema,
    AgentStepSchema,
    DecomposePlanRequest,
    DecomposePlanResponse,
    ExecuteToolRequest,
    ExecuteToolResponse,
    ListToolsResponse,
    ToolDefinitionSchema,
    ToolExecutionResultSchema,
    ToolParameterSchema,
)

router = APIRouter(prefix="/agents", tags=["Multi-Agent Orchestrator & Tool Calling Mesh"])

_tool_registry = ToolRegistry.get_instance()
_supervisor = SupervisorAgent(tool_registry=_tool_registry)


@router.get("/tools", response_model=ListToolsResponse)
async def list_available_tools(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    List all introspected tools registered in the Agent Execution Mesh.
    """
    definitions = _tool_registry.list_definitions()
    tool_schemas: List[ToolDefinitionSchema] = []

    for d in definitions:
        param_schemas = [
            ToolParameterSchema(
                name=p.name,
                type=p.type,
                description=p.description,
                required=p.required,
                default=p.default,
            )
            for p in d.parameters
        ]
        tool_schemas.append(
            ToolDefinitionSchema(
                name=d.name,
                description=d.description,
                category=d.category,
                parameters=param_schemas,
                json_schema=d.to_json_schema(),
            )
        )

    return ListToolsResponse(
        total_tools=len(tool_schemas),
        tools=tool_schemas,
    )


@router.post("/execute-tool", response_model=ExecuteToolResponse)
async def execute_tool_directly(
    request: ExecuteToolRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Directly execute a registered tool with parameter validation.
    """
    res = _tool_registry.execute(tool_name=request.tool_name, arguments=request.arguments)
    return ExecuteToolResponse(
        result=ToolExecutionResultSchema(
            tool_name=res.tool_name,
            arguments=res.arguments,
            output=res.output,
            success=res.success,
            error=res.error,
            latency_ms=res.latency_ms,
        )
    )


@router.post("/plan", response_model=DecomposePlanResponse)
async def decompose_plan(
    request: DecomposePlanRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Decompose a multi-modal user request into an ordered DAG task plan without executing it.
    """
    plan = _supervisor.decompose_plan(query=request.query)

    plan_steps = [
        AgentPlanStepSchema(
            step_id=s.step_id,
            description=s.description,
            assigned_agent=s.assigned_agent,
            dependencies=s.dependencies,
            status=s.status,
            output=s.output,
        )
        for s in plan.steps
    ]

    return DecomposePlanResponse(
        plan=AgentPlanSchema(
            plan_id=plan.plan_id,
            user_intent=plan.user_intent,
            steps=plan_steps,
        )
    )


@router.post("/chat", response_model=AgentChatResponse)
async def agent_chat(
    request: AgentChatRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Execute autonomous multi-agent ReAct orchestration on input query.
    """
    agent_response = _supervisor.run(query=request.query, context=request.context)

    plan_schema = None
    if agent_response.plan:
        plan_steps = [
            AgentPlanStepSchema(
                step_id=s.step_id,
                description=s.description,
                assigned_agent=s.assigned_agent,
                dependencies=s.dependencies,
                status=s.status,
                output=s.output,
            )
            for s in agent_response.plan.steps
        ]
        plan_schema = AgentPlanSchema(
            plan_id=agent_response.plan.plan_id,
            user_intent=agent_response.plan.user_intent,
            steps=plan_steps,
        )

    steps_schemas = [
        AgentStepSchema(
            step_index=s.step_index,
            thought=s.thought,
            action=s.action.model_dump() if s.action else None,
            observation=s.observation.model_dump() if s.observation else None,
            timestamp=s.timestamp,
        )
        for s in agent_response.steps
    ]

    tool_call_schemas = [
        ToolExecutionResultSchema(
            tool_name=t.tool_name,
            arguments=t.arguments,
            output=t.output,
            success=t.success,
            error=t.error,
            latency_ms=t.latency_ms,
        )
        for t in agent_response.tool_calls
    ]

    return AgentChatResponse(
        query=agent_response.query,
        final_answer=agent_response.final_answer,
        plan=plan_schema,
        steps=steps_schemas,
        tool_calls=tool_call_schemas,
        latency_ms=agent_response.latency_ms,
        status=agent_response.status,
    )
