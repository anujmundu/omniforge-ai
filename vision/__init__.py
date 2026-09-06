"""
OmniForge Computer Vision and Video Analytics Package.
"""

from vision.base import (
    BaseDetector,
    BaseOCR,
    BaseTracker,
    BoundingBox,
    Detection,
    DetectionResult,
    OCRResult,
    OCRSpan,
    TrackedObject,
    TrackingResult,
)
from vision.detector import ObjectDetector
from vision.ocr import SpatialOCREngine
from vision.stream import VideoStreamProcessor
from vision.tracker import MultiObjectTracker

__all__ = [
    "BaseDetector",
    "BaseTracker",
    "BaseOCR",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "TrackedObject",
    "TrackingResult",
    "OCRSpan",
    "OCRResult",
    "ObjectDetector",
    "MultiObjectTracker",
    "SpatialOCREngine",
    "VideoStreamProcessor",
]
