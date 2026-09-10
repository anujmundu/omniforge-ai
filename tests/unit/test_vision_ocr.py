"""
Unit tests for Spatial OCR Engine.
"""

import numpy as np

from vision.ocr import SpatialOCREngine


def test_spatial_ocr_extraction():
    ocr = SpatialOCREngine()
    dummy_img = np.zeros((600, 800, 3), dtype=np.uint8)

    res = ocr.extract_text(dummy_img, min_confidence=0.5)

    assert len(res.spans) > 0
    assert "OMNIFORGE" in res.full_text
    assert res.image_width == 800
    assert res.image_height == 600
    assert res.inference_latency_ms >= 0.0

    for span in res.spans:
        assert span.confidence >= 0.5
        assert span.box.xmin < span.box.xmax
        assert span.box.ymin < span.box.ymax
        if span.polygon:
            assert len(span.polygon) == 4
