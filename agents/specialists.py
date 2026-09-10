"""
Domain Specialist Agents for OmniForge Multi-Agent Intelligence Mesh.
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

from agents.base import (
    AgentAction,
    AgentObservation,
    AgentResponse,
    AgentRole,
    AgentStep,
    BaseAgent,
)
from agents.tools import ToolRegistry


class MLAnalyticsAgent(BaseAgent):
    """Specialist agent for tabular ML training, prediction, anomaly detection, and forecasting."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None) -> None:
        super().__init__(
            name="MLAnalyticsAgent",
            role=AgentRole.ML_SPECIALIST,
            description="Executes Classical ML inference, churn classification, regression, and forecasting.",
        )
        self.tools = tool_registry or ToolRegistry.get_instance()

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        start_time = time.perf_counter()
        q_lower = query.lower()

        model_type = "classification"
        if "regr" in q_lower or "price" in q_lower or "cost" in q_lower:
            model_type = "regression"
        elif "anom" in q_lower or "fraud" in q_lower or "outlier" in q_lower:
            model_type = "anomaly"
        elif "fore" in q_lower or "demand" in q_lower or "trend" in q_lower:
            model_type = "forecasting"

        features = (context or {}).get("features", [{"monthly_charges": 85.0, "sqft": 2000, "amount": 6200.0}])

        # Step 1: Reason and dispatch tool call
        thought = f"Identified ML task as '{model_type}'. Invoking ml_predict tool with provided features."
        action = AgentAction(tool_name="ml_predict", arguments={"model_type": model_type, "features": features})
        tool_res = self.tools.execute("ml_predict", action.arguments)

        obs = AgentObservation(output=tool_res.output, error=tool_res.error, success=tool_res.success)
        step = AgentStep(step_index=1, thought=thought, action=action, observation=obs)

        # Step 2: Formulate domain answer
        if model_type == "classification":
            ans = f"ML Classification completed: prediction is Class {tool_res.output.get('prediction')} with confidence {tool_res.output.get('confidence')}."
        elif model_type == "regression":
            ans = f"ML Regression completed: estimated value is ${tool_res.output.get('predicted_values')[0]:,.2f} (R2={tool_res.output.get('r2_score')})."
        elif model_type == "anomaly":
            ans = f"Anomaly Detection completed: Outlier detected={tool_res.output.get('anomaly_detected')} with score {tool_res.output.get('anomaly_scores')}."
        else:
            ans = f"Time-Series Forecasting completed: 7-day trajectory is {tool_res.output.get('horizon_forecast')} with an {tool_res.output.get('trend')} trend."

        latency = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            query=query,
            final_answer=ans,
            steps=[step],
            tool_calls=[tool_res],
            latency_ms=round(latency, 2),
            status="COMPLETED",
        )


class VisionAnalyticsAgent(BaseAgent):
    """Specialist agent for Object Detection, Video Multi-Object Tracking, and Spatial OCR."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None) -> None:
        super().__init__(
            name="VisionAnalyticsAgent",
            role=AgentRole.VISION_SPECIALIST,
            description="Performs visual perception, object detection, spatial OCR layout analysis, and tracking.",
        )
        self.tools = tool_registry or ToolRegistry.get_instance()

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        start_time = time.perf_counter()
        q_lower = query.lower()

        if "ocr" in q_lower or "invoice" in q_lower or "text" in q_lower or "document" in q_lower:
            thought = "Visual document analysis requested. Invoking vision_ocr_extract tool."
            action = AgentAction(
                tool_name="vision_ocr_extract",
                arguments={"document_uri": (context or {}).get("uri", "doc://invoice_01.png")},
            )
            tool_res = self.tools.execute("vision_ocr_extract", action.arguments)
            obs = AgentObservation(output=tool_res.output, error=tool_res.error, success=tool_res.success)
            step = AgentStep(step_index=1, thought=thought, action=action, observation=obs)
            ans = f"Spatial OCR Extraction complete: Document '{tool_res.output.get('structured_fields', {}).get('document_id')}' extracted with amount ${tool_res.output.get('structured_fields', {}).get('total_amount_usd'):,.2f}."
        else:
            thought = "Object detection requested. Invoking vision_detect_objects tool."
            action = AgentAction(
                tool_name="vision_detect_objects",
                arguments={"image_uri": (context or {}).get("uri", "img://frame_01.jpg")},
            )
            tool_res = self.tools.execute("vision_detect_objects", action.arguments)
            obs = AgentObservation(output=tool_res.output, error=tool_res.error, success=tool_res.success)
            step = AgentStep(step_index=1, thought=thought, action=action, observation=obs)
            ans = f"Object Detection complete: {tool_res.output.get('total_detections')} objects detected (classes: {[d['class_name'] for d in tool_res.output.get('detections', [])]})."

        latency = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            query=query,
            final_answer=ans,
            steps=[step],
            tool_calls=[tool_res],
            latency_ms=round(latency, 2),
            status="COMPLETED",
        )


class NLPProcessingAgent(BaseAgent):
    """Specialist agent for Named Entity Recognition, Text Embeddings, and Classification."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None) -> None:
        super().__init__(
            name="NLPProcessingAgent",
            role=AgentRole.NLP_SPECIALIST,
            description="Extracts localized named entities, generates embeddings, and classifies text.",
        )
        self.tools = tool_registry or ToolRegistry.get_instance()

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        start_time = time.perf_counter()
        target_text = (context or {}).get("text", query)

        thought = "Extracting localized named entities and key phrases with exact character spans."
        action = AgentAction(tool_name="nlp_extract_entities", arguments={"text": target_text, "min_confidence": 0.5})
        tool_res = self.tools.execute("nlp_extract_entities", action.arguments)

        obs = AgentObservation(output=tool_res.output, error=tool_res.error, success=tool_res.success)
        step = AgentStep(step_index=1, thought=thought, action=action, observation=obs)

        entities_list = tool_res.output.get("entities", [])
        labels_summary = ", ".join([f"{e['text']} ({e['label']})" for e in entities_list[:5]])
        ans = f"NLP Entity Extraction complete: Found {len(entities_list)} entities: {labels_summary}."

        latency = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            query=query,
            final_answer=ans,
            steps=[step],
            tool_calls=[tool_res],
            latency_ms=round(latency, 2),
            status="COMPLETED",
        )


class EnterpriseRAGAgent(BaseAgent):
    """Specialist agent for Knowledge Base semantic search and grounded citation retrieval."""

    def __init__(self, tool_registry: Optional[ToolRegistry] = None) -> None:
        super().__init__(
            name="EnterpriseRAGAgent",
            role=AgentRole.RAG_SPECIALIST,
            description="Searches enterprise knowledge base collections and synthesizes citation-grounded evidence.",
        )
        self.tools = tool_registry or ToolRegistry.get_instance()

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        start_time = time.perf_counter()
        coll = (context or {}).get("collection_name", "enterprise_kb")

        thought = f"Performing hybrid vector retrieval and reranking on collection '{coll}'."
        action = AgentAction(tool_name="rag_search_knowledge_base", arguments={"query": query, "collection_name": coll})
        tool_res = self.tools.execute("rag_search_knowledge_base", action.arguments)

        obs = AgentObservation(output=tool_res.output, error=tool_res.error, success=tool_res.success)
        step = AgentStep(step_index=1, thought=thought, action=action, observation=obs)

        results = tool_res.output.get("results", [])
        snippets = " ".join([f"[{i+1}] {r['chunk_text']}" for i, r in enumerate(results)])
        ans = f"RAG Knowledge Base Search complete: Retrieved {len(results)} verified passages: {snippets}"

        latency = (time.perf_counter() - start_time) * 1000.0

        return AgentResponse(
            query=query,
            final_answer=ans,
            steps=[step],
            tool_calls=[tool_res],
            latency_ms=round(latency, 2),
            status="COMPLETED",
        )
