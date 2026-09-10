"""
Vision API Router for Object Detection, Video Tracking, and Spatial OCR.
"""

from __future__ import annotations

import base64
import time
from typing import Any, List

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.core.dependencies import get_current_user
from apps.api.models.user import User
from apps.api.schemas.vision import (
    BoundingBoxSchema,
    DetectImageRequest,
    DetectImageResponse,
    DetectionItemSchema,
    OCRRequest,
    OCRResponse,
    OCRSpanSchema,
    TrackedObjectSchema,
    TrackFrameResponse,
    VideoTrackingRequest,
    VideoTrackingResponse,
    VisionModelInfoResponse,
)
from vision.detector import ObjectDetector
from vision.ocr import SpatialOCREngine
from vision.tracker import MultiObjectTracker

router = APIRouter(prefix="/vision", tags=["Computer Vision & Video Analytics"])

# Global in-memory singleton instances for high-throughput serving
_detector = ObjectDetector(model_name="yolov8n", backend="simulated")
_tracker = MultiObjectTracker(max_age=30, min_hits=1, iou_threshold=0.30)
_ocr_engine = SpatialOCREngine(engine_type="spatial_hybrid")


def _decode_base64_or_dummy(image_b64: str | None) -> bytes:
    """Helper to decode base64 string or provide fallback synthetic image bytes."""
    if not image_b64:
        # 100x100 RGB dummy byte array encoded in PNG format
        import io

        from PIL import Image

        img = Image.new("RGB", (640, 480), color=(73, 109, 137))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    try:
        # Strip data URL prefix if present
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]
        return base64.b64decode(image_b64)
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid base64 image data: {str(exc)}")


@router.get("/models", response_model=VisionModelInfoResponse)
async def get_vision_models(
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    List available vision models, architectures, and supported classes.
    """
    return VisionModelInfoResponse(
        models=[
            {
                "model_id": "yolov8n",
                "task": "object_detection",
                "backend": _detector.backend,
                "input_resolution": "640x640",
                "parameters_count": "3.2M",
                "quantization": "FP16 / INT8",
            },
            {
                "model_id": "bytetrack_spatial",
                "task": "multi_object_tracking",
                "association_metric": "IoU + Centroid",
                "max_age": _tracker.max_age,
            },
            {
                "model_id": "spatial_ocr_hybrid",
                "task": "optical_character_recognition",
                "geometry": "4-point polygon + bounding box",
                "supported_languages": ["en", "es", "fr", "de"],
            },
        ],
        supported_classes=_detector.COCO_CLASSES,
    )


@router.post("/detect", response_model=DetectImageResponse)
async def detect_objects(
    request: DetectImageRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Run object detection on an image input.
    Returns normalized bounding boxes, confidence scores, and class labels.
    """
    image_bytes = _decode_base64_or_dummy(request.image_base64)

    det_result = _detector.detect(
        image=image_bytes,
        confidence_threshold=request.confidence_threshold,
        iou_threshold=request.iou_threshold,
        classes=request.classes,
    )

    detections = [
        DetectionItemSchema(
            label=d.label,
            confidence=d.confidence,
            box=BoundingBoxSchema(
                xmin=d.box.xmin,
                ymin=d.box.ymin,
                xmax=d.box.xmax,
                ymax=d.box.ymax,
                pixel_box=d.box.pixel_box,
            ),
            class_id=d.class_id,
            metadata=d.metadata,
        )
        for d in det_result.detections
    ]

    return DetectImageResponse(
        image_width=det_result.image_width,
        image_height=det_result.image_height,
        total_detections=len(detections),
        detections=detections,
        inference_latency_ms=det_result.inference_latency_ms,
        model_name=det_result.model_name,
    )


@router.post("/ocr", response_model=OCRResponse)
async def extract_spatial_ocr(
    request: OCRRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Extract spatial text spans, confidence scores, and polygon bounding boxes.
    """
    image_bytes = _decode_base64_or_dummy(request.image_base64)
    ocr_result = _ocr_engine.extract_text(
        image=image_bytes,
        min_confidence=request.min_confidence,
        language=request.language,
    )

    spans = [
        OCRSpanSchema(
            text=s.text,
            confidence=s.confidence,
            box=BoundingBoxSchema(
                xmin=s.box.xmin,
                ymin=s.box.ymin,
                xmax=s.box.xmax,
                ymax=s.box.ymax,
                pixel_box=s.box.pixel_box,
            ),
            polygon=s.polygon,
        )
        for s in ocr_result.spans
    ]

    return OCRResponse(
        full_text=ocr_result.full_text,
        total_spans=len(spans),
        spans=spans,
        image_width=ocr_result.image_width,
        image_height=ocr_result.image_height,
        language=ocr_result.language or "en",
        inference_latency_ms=ocr_result.inference_latency_ms,
    )


@router.post("/track/video", response_model=VideoTrackingResponse)
async def track_video_frames(
    request: VideoTrackingRequest,
    current_user: User = Depends(get_current_user),
) -> Any:
    """
    Run multi-object tracking across an ordered sequence of video frames.
    Returns persistent track IDs, centroid history, and estimated velocity vectors.
    """
    start_total = time.perf_counter()
    _tracker.reset()

    frame_results: List[TrackFrameResponse] = []
    interval_ms = 1000.0 / request.fps

    for idx, b64_frame in enumerate(request.frames_base64):
        frame_bytes = _decode_base64_or_dummy(b64_frame)
        ts_ms = round(idx * interval_ms, 2)

        # Detection
        det_result = _detector.detect(
            image=frame_bytes,
            confidence_threshold=request.confidence_threshold,
            classes=request.classes,
            frame_index=idx,
            timestamp_ms=ts_ms,
        )

        # Tracking update
        track_result = _tracker.update(
            detections=det_result.detections,
            frame_index=idx,
            timestamp_ms=ts_ms,
        )

        active_tracks = [
            TrackedObjectSchema(
                track_id=t.track_id,
                label=t.label,
                confidence=t.confidence,
                current_box=BoundingBoxSchema(
                    xmin=t.current_box.xmin,
                    ymin=t.current_box.ymin,
                    xmax=t.current_box.xmax,
                    ymax=t.current_box.ymax,
                    pixel_box=t.current_box.pixel_box,
                ),
                history_centers=t.history_centers,
                age_frames=t.age_frames,
                velocity=t.velocity,
            )
            for t in track_result.active_tracks
        ]

        frame_results.append(
            TrackFrameResponse(
                frame_index=idx,
                timestamp_ms=ts_ms,
                active_tracks=active_tracks,
                total_tracks_observed=track_result.total_tracks_observed,
                inference_latency_ms=round(det_result.inference_latency_ms + track_result.inference_latency_ms, 2),
            )
        )

    total_latency = (time.perf_counter() - start_total) * 1000.0

    return VideoTrackingResponse(
        total_frames_processed=len(request.frames_base64),
        total_unique_tracks=_tracker._total_tracks_count,
        frame_results=frame_results,
        total_latency_ms=round(total_latency, 2),
    )
