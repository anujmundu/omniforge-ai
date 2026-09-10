"""
Base contracts and domain models for Computer Vision and Video Analytics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """Normalized bounding box coordinates in [0.0, 1.0] space and pixel space."""

    # Normalized coordinates [0.0, 1.0]
    xmin: float = Field(..., ge=0.0, le=1.0, description="Normalized top-left X")
    ymin: float = Field(..., ge=0.0, le=1.0, description="Normalized top-left Y")
    xmax: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom-right X")
    ymax: float = Field(..., ge=0.0, le=1.0, description="Normalized bottom-right Y")

    # Absolute pixel dimensions (optional if image shape is known)
    pixel_box: Optional[Tuple[int, int, int, int]] = Field(
        default=None, description="Absolute pixel coordinates (x1, y1, x2, y2)"
    )

    @property
    def width(self) -> float:
        return max(0.0, self.xmax - self.xmin)

    @property
    def height(self) -> float:
        return max(0.0, self.ymax - self.ymin)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def to_pixel(self, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """Convert normalized coordinates to absolute integer pixel coordinates."""
        x1 = int(round(self.xmin * img_width))
        y1 = int(round(self.ymin * img_height))
        x2 = int(round(self.xmax * img_width))
        y2 = int(round(self.ymax * img_height))
        return (max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2))

    def iou(self, other: BoundingBox) -> float:
        """Calculate Intersection over Union (IoU) with another bounding box."""
        ixmin = max(self.xmin, other.xmin)
        iymin = max(self.ymin, other.ymin)
        ixmax = min(self.xmax, other.xmax)
        iymax = min(self.ymax, other.ymax)

        iw = max(0.0, ixmax - ixmin)
        ih = max(0.0, iymax - iymin)
        intersection = iw * ih

        union = self.area + other.area - intersection
        if union <= 0.0:
            return 0.0
        return intersection / union


class Detection(BaseModel):
    """Single object detection instance."""

    label: str = Field(..., description="Class name / label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    box: BoundingBox = Field(..., description="Spatial bounding box")
    class_id: Optional[int] = Field(default=None, description="Numeric class identifier")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Auxiliary detection metadata")


class DetectionResult(BaseModel):
    """Complete detection output for an image frame."""

    frame_index: int = Field(default=0, description="Frame index in stream/batch")
    timestamp_ms: Optional[float] = Field(default=None, description="Frame timestamp in milliseconds")
    image_width: int = Field(..., description="Image width in pixels")
    image_height: int = Field(..., description="Image height in pixels")
    detections: List[Detection] = Field(default_factory=list, description="List of detected objects")
    inference_latency_ms: float = Field(default=0.0, description="Inference latency in milliseconds")
    model_name: str = Field(default="yolo_detector", description="Model identifier used for inference")

    @property
    def count(self) -> int:
        return len(self.detections)


class TrackedObject(BaseModel):
    """Object tracked across sequential video frames with persistent ID and history."""

    track_id: int = Field(..., description="Unique persistent tracking identifier")
    label: str = Field(..., description="Class label")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Current confidence score")
    current_box: BoundingBox = Field(..., description="Current frame bounding box")
    history_centers: List[Tuple[float, float]] = Field(
        default_factory=list, description="Historical centroid trajectory [(x, y), ...]"
    )
    age_frames: int = Field(default=1, description="Number of frames this object has been tracked")
    time_since_update: int = Field(default=0, description="Frames since object was last matched")
    velocity: Optional[Tuple[float, float]] = Field(
        default=(0.0, 0.0), description="Estimated velocity vector (dx/frame, dy/frame)"
    )


class TrackingResult(BaseModel):
    """Multi-object tracking output for a video frame."""

    frame_index: int = Field(..., description="Video frame sequence index")
    timestamp_ms: Optional[float] = Field(default=None, description="Frame timestamp in milliseconds")
    active_tracks: List[TrackedObject] = Field(default_factory=list, description="Currently active tracked objects")
    total_tracks_observed: int = Field(default=0, description="Cumulative unique objects observed")
    inference_latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")


class OCRSpan(BaseModel):
    """Individual text snippet extracted via OCR with spatial localization."""

    text: str = Field(..., description="Extracted text string")
    confidence: float = Field(..., ge=0.0, le=1.0, description="OCR confidence score")
    box: BoundingBox = Field(..., description="Bounding box containing the text")
    polygon: Optional[List[Tuple[float, float]]] = Field(
        default=None, description="Detailed polygon coordinates [(x1, y1), (x2, y2), ...]"
    )


class OCRResult(BaseModel):
    """Complete OCR extraction output."""

    full_text: str = Field(..., description="Aggregated readable text extracted from document/scene")
    spans: List[OCRSpan] = Field(default_factory=list, description="List of localized text spans")
    image_width: int = Field(..., description="Source image width")
    image_height: int = Field(..., description="Source image height")
    language: Optional[str] = Field(default="en", description="Detected or requested language code")
    inference_latency_ms: float = Field(default=0.0, description="Processing latency in milliseconds")


# ==============================================================================
# Abstract Engine Contracts
# ==============================================================================


class BaseDetector(ABC):
    """Abstract interface for object detection models."""

    @abstractmethod
    def detect(
        self,
        image: Any,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        classes: Optional[List[str]] = None,
    ) -> DetectionResult:
        """Run object detection on an image input (numpy array, PIL Image, or bytes)."""
        pass


class BaseTracker(ABC):
    """Abstract interface for multi-object tracking across video streams."""

    @abstractmethod
    def update(
        self,
        detections: List[Detection],
        frame_index: int,
        timestamp_ms: Optional[float] = None,
    ) -> TrackingResult:
        """Update tracker state with new frame detections and return active tracks."""
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset internal tracker state for a new video sequence."""
        pass


class BaseOCR(ABC):
    """Abstract interface for Optical Character Recognition engines."""

    @abstractmethod
    def extract_text(
        self,
        image: Any,
        min_confidence: float = 0.30,
        language: str = "en",
    ) -> OCRResult:
        """Extract localized text spans from an image."""
        pass
