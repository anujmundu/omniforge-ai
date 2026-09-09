# ADR-014: Declarative Tool Calling & Agent Execution Mesh

## Status
Accepted

## Date
2026-09-09

## Context
Autonomous agents require structured mechanisms to interact with computational tools, model inference engines, database stores, and external APIs. Without a unified tool abstraction, tool calling becomes brittle, error-prone, untyped, and vulnerable to arbitrary execution flaws.

We required an enterprise tool execution mesh that satisfies four criteria:
1. **Declarative Decorator Syntax**: Zero boilerplate for exposing Python functions as agent tools.
2. **Strict Schema Introspection**: Automatic derivation of Pydantic v2 and JSON Schema definitions (parameter names, types, default values, docstrings).
3. **Execution Safety & Parameter Validation**: Runtime validation of tool arguments against schema definitions before execution.
4. **Audit Logging & Structured Execution Traces**: Detailed capturing of tool execution latency, caller identity, arguments, return status, and exception traces.

## Decision
We implemented a **Declarative Tool Execution Framework** centered around the `@tool` decorator and a centralized `ToolRegistry`:

1. **`@tool` Decorator**:
   - Inspects function signatures using Python's `inspect` module and type annotations.
   - Extracts descriptive docstrings for tool description and parameter semantics.
   - Generates compliant JSON Schema specification dictionaries compatible with OpenAI / Anthropic / Local LLM tool-calling conventions.

2. **`ToolRegistry` Singleton & Namespacing**:
   - Central registry indexing tools by unique identifiers (e.g. `ml_predict`, `vision_detect_objects`, `nlp_extract_entities`, `rag_search_knowledge_base`, `sql_execute_query`).
   - Supports domain namespacing and access-control filtering per specialist agent.

3. **Runtime Parameter Validation & Error Handling**:
   - Automatically casts and validates incoming dictionary payloads against declared types.
   - Intercepts runtime exceptions, wrapping them into structured `ToolExecutionResult(success=False, error=str(e))` payloads that allow the ReAct agent to observe the failure and self-correct.

4. **Built-in Standard Tool Library**:
   - **`ml_predict`**: Executes real-time inference on registered Classical ML models.
   - **`vision_detect_objects`**: Runs YOLO object detection on simulated or real image arrays.
   - **`vision_ocr_extract`**: Runs Spatial OCR layout extraction.
   - **`nlp_extract_entities`**: Extracts localized named entities with span offsets.
   - **`rag_search_knowledge_base`**: Executes hybrid vector recall and cross-encoder reranking.
   - **`sql_execute_query`**: Runs validated analytical SQL queries against the metadata store.

## Consequences

### Positive
- **Type Safety**: Invalid parameters are rejected immediately before invocation, preventing runtime model crashes.
- **Zero-Friction Tool Creation**: Adding a new tool requires only decorating standard Python functions with `@tool(name="...", description="...")`.
- **Extensibility**: Agents can dynamically introspect available tools and schemas via the `GET /api/v1/agents/tools` REST API endpoint.

### Trade-offs
- Automatic schema generation requires disciplined type annotations and docstring formatting across all registered functions.
