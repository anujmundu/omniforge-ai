# ADR 008: Video Stream Processing and Spatial OCR Pipeline

## Status
Accepted

## Date
2026-09-06

## Context
High-volume multimodal AI platforms must ingest high-framerate video streams and unstructured document/scene images containing localized text without blocking the main event loop or incurring unbounded memory spikes.

## Decision
1. **Async Video Stream Ingestion (`vision/stream.py`)**:
   - Generator-based frame decoding with backpressure-aware ring buffers.
   - Configurable frame-skip rate (`stride`) to optimize inference compute vs. temporal resolution.
   - Batch frame dispatch to parallel or GPU-accelerated inference pipelines.
2. **Spatial OCR Text Extraction (`vision/ocr.py`)**:
   - Bounding polygon coordinates $[x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4]$ for tilted/skewed text.
   - Word-level and line-level confidence scoring.
   - Text post-processing filters (regex, alphanumeric cleaning, and entity masking).

## Consequences
- High-efficiency frame buffering preventing memory exhaustion during 4K video feeds.
- Full support for mixed vision workloads (simultaneous object detection + OCR overlay).
