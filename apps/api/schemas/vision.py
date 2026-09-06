"""
Pydantic schemas for Computer Vision and Video Analytics REST APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    """Normalized spatial coordinates [0.0, 1.0]."""
    xmin: float = Field(..., ge=0.0, le=1.0)
    ymin: float = Field(..., ge=0.0, le=1.0)
    xmax: float = Field(..., ge=0.0, le=1.0)
    ymax: float = Field(..., ge=0.0, le=1.0)
    pixel_box: Optional[Tuple[int, int, int, int]] = None


class DetectionItemSchema(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: BoundingBoxSchema
    class_id: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DetectImageRequest(BaseModel):
    """Request payload for image object detection."""
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image string")
    image_url: Optional[str] = Field(default=None, description="Public image URL")
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    classes: Optional[List[str]] = Field(default=None, description="Optional class allowlist filter")


class DetectImageResponse(BaseModel):
    """Response payload for image object detection."""
    image_width: int
    image_height: int
    total_detections: int
    detections: List[DetectionItemSchema]
    inference_latency_ms: float
    model_name: str


class OCRSpanSchema(BaseModel):
    text: str
    confidence: float
    box: BoundingBoxSchema
    polygon: Optional[List[Tuple[float, float]]] = None


class OCRRequest(BaseModel):
    """Request payload for OCR extraction."""
    image_base64: Optional[str] = Field(default=None, description="Base64-encoded image string")
    min_confidence: float = Field(default=0.30, ge=0.0, le=1.0)
    language: str = Field(default="en")


class OCRResponse(BaseModel):
    """Response payload for OCR extraction."""
    full_text: str
    total_spans: int
    spans: List[OCRSpanSchema]
    image_width: int
    image_height: int
    language: str
    inference_latency_ms: float


class TrackedObjectSchema(BaseModel):
    track_id: int
    label: str
    confidence: float
    current_box: BoundingBoxSchema
    history_centers: List[Tuple[float, float]]
    age_frames: int
    velocity: Optional[Tuple[float, float]] = None


class TrackFrameResponse(BaseModel):
    frame_index: int
    timestamp_ms: Optional[float] = None
    active_tracks: List[TrackedObjectSchema]
    total_tracks_observed: int
    inference_latency_ms: float


class VideoTrackingRequest(BaseModel):
    """Request payload for multi-frame video tracking."""
    frames_base64: List[str] = Field(..., min_length=1, description="Ordered list of base64 video frames")
    fps: float = Field(default=30.0, ge=1.0)
    confidence_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    classes: Optional[List[str]] = None


class VideoTrackingResponse(BaseModel):
    total_frames_processed: int
    total_unique_tracks: int
    frame_results: List[TrackFrameResponse]
    total_latency_ms: float


class VisionModelInfoResponse(BaseModel):
    models: List[Dict[str, Any]]
    supported_classes: List[str]
