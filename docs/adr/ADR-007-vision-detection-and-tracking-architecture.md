# ADR 007: Computer Vision Object Detection and Spatial Tracking Architecture

## Status
Accepted

## Date
2026-09-06

## Context
OmniForge requires high-throughput computer vision capabilities across static images and dynamic video sequences. Key operational requirements include:
1. Standardized object detection output format with normalized $[x_{min}, y_{min}, x_{max}, y_{max}]$ coordinates (0.0 to 1.0) and pixel absolute bounds.
2. Multi-object tracking (MOT) across consecutive video frames that maintains persistent trajectory IDs under occlusion, frame drops, and rapid spatial movements.
3. Decoupled detector backends (support for Ultralytics YOLOv8/v11, OpenCV DNN, and lightweight edge inference pipelines).
4. Sub-50ms inference latency per frame on standard compute instances.

## Decision
We implement a layered vision architecture in `vision/`:
1. **Contracts (`vision/base.py`)**: Abstract `BaseDetector` and `BaseTracker` enforcing uniform `detect()` and `track()` interfaces returning typed `DetectionBox`, `DetectionResult`, and `TrackedObject` models.
2. **Object Detection (`vision/detector.py`)**: Unified detector class with Non-Maximum Suppression (NMS), confidence filtering, class allowlisting, and coordinate normalization.
3. **Multi-Object Spatial Tracker (`vision/tracker.py`)**: Centroid and Intersection-over-Union (IoU) spatial association tracker with Kalman filter state estimation, max-age disappearance handling, and historical trajectory recording.

## Consequences
- Single unified API contract regardless of underlying model architecture.
- Full compatibility with image batches, video arrays, and streaming byte frames.
- Enables downstream modules (e.g. video analytics, autonomous surveillance, retail intelligence) to consume persistent track IDs.
