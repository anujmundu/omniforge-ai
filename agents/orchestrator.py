"""
Hierarchical Supervisor Agent and Multi-Agent Execution Orchestrator.
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Optional

from agents.base import (
    AgentAction,
    AgentObservation,
    AgentPlan,
    AgentPlanStep,
    AgentResponse,
    AgentRole,
    AgentStep,
    BaseAgent,
    ToolExecutionResult,
)
from agents.memory import AgentMemory
from agents.specialists import (
    EnterpriseRAGAgent,
    MLAnalyticsAgent,
    NLPProcessingAgent,
    VisionAnalyticsAgent,
)
from agents.tools import ToolRegistry


class SupervisorAgent(BaseAgent):
    """
    Supervisor / Orchestrator Agent.
    Decomposes multi-modal requests into structured task graphs (DAG),
    delegates sub-goals to domain specialists, coordinates ReAct loops,
    and synthesizes comprehensive answers.
    """

    def __init__(
        self,
        memory: Optional[AgentMemory] = None,
        tool_registry: Optional[ToolRegistry] = None,
    ) -> None:
        super().__init__(
            name="OmniForgeSupervisor",
            role=AgentRole.SUPERVISOR,
            description="Autonomous Multi-Agent Planner and Orchestrator.",
        )
        self.memory = memory or AgentMemory()
        self.tools = tool_registry or ToolRegistry.get_instance()

        # Domain Specialist Agents
        self.ml_specialist = MLAnalyticsAgent(self.tools)
        self.vision_specialist = VisionAnalyticsAgent(self.tools)
        self.nlp_specialist = NLPProcessingAgent(self.tools)
        self.rag_specialist = EnterpriseRAGAgent(self.tools)

    def decompose_plan(self, query: str) -> AgentPlan:
        """
        Analyze prompt and decompose into dependency-ordered execution steps.
        """
        q_lower = query.lower()
        steps: List[AgentPlanStep] = []
        step_id = 1

        # 1. Visual / Document OCR
        if any(w in q_lower for w in ["image", "video", "ocr", "invoice", "detect", "track", "camera"]):
            desc = "Extract visual entities, bounding boxes, or spatial document text"
            steps.append(
                AgentPlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent="VisionAnalyticsAgent",
                    dependencies=[],
                )
            )
            step_id += 1

        # 2. NLP Entity Extraction / Embeddings
        if any(w in q_lower for w in ["entity", "ner", "sentiment", "nlp", "extract", "names", "spans"]):
            deps = [1] if steps else []
            desc = "Extract named entity spans and semantic classifications"
            steps.append(
                AgentPlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent="NLPProcessingAgent",
                    dependencies=deps,
                )
            )
            step_id += 1

        # 3. Enterprise Knowledge Base RAG
        if any(
            w in q_lower
            for w in ["knowledge", "doc", "rag", "policy", "search", "retrieve", "guide", "manual", "explain"]
        ):
            desc = "Search enterprise knowledge base for verified documentation and citations"
            steps.append(
                AgentPlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent="EnterpriseRAGAgent",
                    dependencies=[],
                )
            )
            step_id += 1

        # 4. Classical ML Predictive Analytics
        if any(
            w in q_lower for w in ["predict", "churn", "price", "forecast", "anomaly", "fraud", "estimate", "trend"]
        ):
            deps = [s.step_id for s in steps]
            desc = "Execute Classical ML inference model (classification / regression / anomaly / forecast)"
            steps.append(
                AgentPlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent="MLAnalyticsAgent",
                    dependencies=deps,
                )
            )
            step_id += 1

        # 5. Database Analytics SQL
        if any(w in q_lower for w in ["sql", "database", "query", "table", "stats", "utilization", "department"]):
            desc = "Execute aggregate SQL analytical query"
            steps.append(
                AgentPlanStep(
                    step_id=step_id,
                    description=desc,
                    assigned_agent="SQLTool",
                    dependencies=[],
                )
            )
            step_id += 1

        # Fallback default step if general query
        if not steps:
            steps.append(
                AgentPlanStep(
                    step_id=1,
                    description="Consult enterprise knowledge base and domain agents",
                    assigned_agent="EnterpriseRAGAgent",
                    dependencies=[],
                )
            )

        return AgentPlan(
            plan_id=f"plan_{uuid.uuid4().hex[:8]}",
            user_intent=query,
            steps=steps,
        )

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """
        Execute full multi-agent ReAct orchestration workflow.
        """
        start_time = time.perf_counter()
        self.memory.add_message(role="user", content=query)

        # 1. Decompose into Plan DAG
        plan = self.decompose_plan(query)
        all_steps: List[AgentStep] = []
        all_tool_calls: List[ToolExecutionResult] = []
        step_results: Dict[int, str] = {}

        # 2. Execute Plan Steps sequentially respecting dependencies
        for p_step in plan.steps:
            p_step.status = "IN_PROGRESS"
            agent_name = p_step.assigned_agent

            thought = f"Executing plan step #{p_step.step_id} ('{p_step.description}') assigned to {agent_name}."

            if agent_name == "VisionAnalyticsAgent":
                sub_res = self.vision_specialist.run(query=query, context=context)
                p_step.output = sub_res.final_answer
                p_step.status = "COMPLETED"
                all_tool_calls.extend(sub_res.tool_calls)
                step_results[p_step.step_id] = sub_res.final_answer

                action = AgentAction(tool_name="vision_analysis", arguments={"query": query})
                obs = AgentObservation(output=sub_res.final_answer, success=True)
                step = AgentStep(step_index=len(all_steps) + 1, thought=thought, action=action, observation=obs)
                all_steps.append(step)

            elif agent_name == "NLPProcessingAgent":
                sub_res = self.nlp_specialist.run(query=query, context=context)
                p_step.output = sub_res.final_answer
                p_step.status = "COMPLETED"
                all_tool_calls.extend(sub_res.tool_calls)
                step_results[p_step.step_id] = sub_res.final_answer

                action = AgentAction(tool_name="nlp_processing", arguments={"query": query})
                obs = AgentObservation(output=sub_res.final_answer, success=True)
                step = AgentStep(step_index=len(all_steps) + 1, thought=thought, action=action, observation=obs)
                all_steps.append(step)

            elif agent_name == "EnterpriseRAGAgent":
                sub_res = self.rag_specialist.run(query=query, context=context)
                p_step.output = sub_res.final_answer
                p_step.status = "COMPLETED"
                all_tool_calls.extend(sub_res.tool_calls)
                step_results[p_step.step_id] = sub_res.final_answer

                action = AgentAction(tool_name="rag_search", arguments={"query": query})
                obs = AgentObservation(output=sub_res.final_answer, success=True)
                step = AgentStep(step_index=len(all_steps) + 1, thought=thought, action=action, observation=obs)
                all_steps.append(step)

            elif agent_name == "MLAnalyticsAgent":
                sub_res = self.ml_specialist.run(query=query, context=context)
                p_step.output = sub_res.final_answer
                p_step.status = "COMPLETED"
                all_tool_calls.extend(sub_res.tool_calls)
                step_results[p_step.step_id] = sub_res.final_answer

                action = AgentAction(tool_name="ml_predict", arguments={"query": query})
                obs = AgentObservation(output=sub_res.final_answer, success=True)
                step = AgentStep(step_index=len(all_steps) + 1, thought=thought, action=action, observation=obs)
                all_steps.append(step)

            elif agent_name == "SQLTool":
                tool_res = self.tools.execute("sql_execute_query", {"query_text": query})
                p_step.output = tool_res.output
                p_step.status = "COMPLETED"
                all_tool_calls.append(tool_res)
                step_results[p_step.step_id] = f"SQL rows returned: {tool_res.output.get('row_count')}"

                action = AgentAction(tool_name="sql_execute_query", arguments={"query_text": query})
                obs = AgentObservation(output=tool_res.output, success=True)
                step = AgentStep(step_index=len(all_steps) + 1, thought=thought, action=action, observation=obs)
                all_steps.append(step)

        # 3. Synthesize Multi-Agent Final Answer
        synthesis_paragraphs = []
        for s_id, s_out in step_results.items():
            synthesis_paragraphs.append(f"• [Step {s_id}]: {s_out}")

        final_ans = (
            f"Multi-Agent Orchestrator successfully executed {len(plan.steps)} coordinated sub-tasks for intent '{query}':\n\n"
            + "\n".join(synthesis_paragraphs)
        )

        latency = (time.perf_counter() - start_time) * 1000.0

        self.memory.add_message(role="assistant", content=final_ans, name=self.name)

        return AgentResponse(
            query=query,
            final_answer=final_ans,
            plan=plan,
            steps=all_steps,
            tool_calls=all_tool_calls,
            latency_ms=round(latency, 2),
            status="COMPLETED",
        )
