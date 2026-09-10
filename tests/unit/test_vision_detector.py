"""
Unit tests for Object Detection engine.
"""

import numpy as np
import pytest

from vision.base import BoundingBox
from vision.detector import ObjectDetector


def test_bounding_box_geometry_and_iou():
    box1 = BoundingBox(xmin=0.1, ymin=0.1, xmax=0.5, ymax=0.5)
    box2 = BoundingBox(xmin=0.3, ymin=0.3, xmax=0.7, ymax=0.7)

    assert box1.width == pytest.approx(0.4)
    assert box1.height == pytest.approx(0.4)
    assert box1.area == pytest.approx(0.16)
    assert box1.center == pytest.approx((0.3, 0.3))

    pixel_coords = box1.to_pixel(img_width=1000, img_height=500)
    assert pixel_coords == (100, 50, 500, 250)

    # IoU calculation
    iou = box1.iou(box2)
    assert 0.0 < iou < 1.0


def test_object_detector_inference_on_array():
    detector = ObjectDetector(model_name="yolov8n", backend="simulated")
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    res = detector.detect(dummy_img, confidence_threshold=0.5)
    assert res.image_width == 640
    assert res.image_height == 480
    assert res.count > 0
    assert res.inference_latency_ms >= 0.0

    for d in res.detections:
        assert 0.0 <= d.box.xmin <= 1.0
        assert 0.0 <= d.box.xmax <= 1.0
        assert d.confidence >= 0.5


def test_object_detector_class_filtering():
    detector = ObjectDetector()
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

    # Allow only 'person'
    res = detector.detect(dummy_img, confidence_threshold=0.1, classes=["person"])
    for d in res.detections:
        assert d.label == "person"
