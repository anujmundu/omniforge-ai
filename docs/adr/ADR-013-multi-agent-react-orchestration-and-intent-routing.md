# ADR-013: Multi-Agent ReAct Orchestration & Intent Routing Architecture

## Status
Accepted

## Date
2026-09-09

## Context
As OmniForge expanded with specialized capabilities across Classical Machine Learning (classification, regression, anomaly detection, forecasting), Computer Vision (detection, tracking, OCR), Natural Language Processing (embeddings, NER, classification), and Enterprise RAG (document ingestion, dense vector search, reranking), single-step linear API routing became insufficient for handling complex, multi-modal user goals.

Real-world enterprise problems frequently require composing heterogeneous modalities into a coordinated workflow (e.g. *"Extract invoice line items with OCR, check supplier history with SQL, retrieve warranty terms from documentation with RAG, and forecast next quarter spend with Time-Series ML"*).

We evaluated three architectural paradigms:
1. **Monolithic Universal Agent**: A single LLM loop given access to dozens of unconstrained tool definitions.
2. **Autonomous Multi-Agent Hierarchical Mesh (Supervisor / Planner + Domain Specialists + ReAct Execution Loop)**: A structured hierarchy where a Supervisor agent decomposes intents into directed task graphs and delegates sub-tasks to focused domain specialist agents operating with localized toolsets.
3. **Static DAG Pipeline**: Hardcoded sequential workflow definitions (e.g., Airflow-style fixed pipelines).

## Decision
We implemented a **Hierarchical Multi-Agent ReAct Architecture** orchestrated by a Supervisor/Planner agent:

1. **Supervisor / Intent Decomposition Agent**:
   - Evaluates user intent and decomposes ambiguous prompts into ordered execution sub-tasks.
   - Assigns sub-tasks to designated specialist agents.
   - Synthesizes intermediate specialist findings into a cohesive, evidence-backed final response.

2. **Domain Specialist Agents**:
   - **`MLAnalyticsAgent`**: Specializes in model training, inference, feature importance, anomaly scoring, and time-series projections.
   - **`VisionAnalyticsAgent`**: Specializes in image/video perception, bounding box geometry, object tracking trajectories, and spatial OCR text layout.
   - **`NLPProcessingAgent`**: Specializes in text vectorization, entity extraction with span validation, and multi-class classification.
   - **`EnterpriseRAGAgent`**: Specializes in vector collections, semantic retrieval, cross-encoder reranking, and citation grounding.

3. **ReAct (Reason + Act) Execution Protocol**:
   - Explicit step-by-step state representation: `Thought` $\rightarrow$ `Action` (Tool Name + Arguments) $\rightarrow$ `Observation` (Execution Output) $\rightarrow$ `Final Answer`.
   - Cycle detection and maximum step limits ($N=8$) to prevent infinite execution loops.

4. **Short-Term Conversational & Working Memory**:
   - Dual-memory design consisting of a FIFO message history window and a dynamic scratchpad holding intermediate tool observations and structured payloads.

## Consequences

### Positive
- **High Modularity & Separation of Concerns**: Each specialist agent operates within a focused domain context and a minimal toolset, reducing hallucination and parameter errors.
- **Traceability & Auditability**: Every step in the multi-agent chain emits structured `AgentStep` records with timestamps, thoughts, tool inputs, and raw outputs.
- **Resilience & Graceful Degradation**: If an individual specialist or tool fails, the ReAct loop catches the error, reflects on the observation, and attempts alternate fallback paths.

### Trade-offs
- Multi-step agent loops introduce cumulative inference latency compared to direct single-endpoint calls (mitigated via streaming traces and optimized in-memory dispatching).
- Requires robust cycle detection and strict iteration caps to ensure predictable execution bounds.
