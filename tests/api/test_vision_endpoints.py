"""
API Integration tests for Vision REST Endpoints (/api/v1/vision/*).
"""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_get_vision_models(client: AsyncClient, admin_headers: dict):
    response = await client.get(
        "/api/v1/vision/models",
        headers=admin_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) >= 3
    assert "person" in data["supported_classes"]


@pytest.mark.asyncio
async def test_detect_objects_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "image_base64": None,  # Will fallback to clean synthetic frame
        "confidence_threshold": 0.25,
        "classes": ["person", "laptop"],
    }
    response = await client.post(
        "/api/v1/vision/detect",
        json=payload,
        headers=engineer_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_detections"] > 0
    assert data["image_width"] > 0
    assert data["image_height"] > 0
    for det in data["detections"]:
        assert det["label"] in ["person", "laptop"]


@pytest.mark.asyncio
async def test_ocr_extraction_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "image_base64": None,
        "min_confidence": 0.30,
        "language": "en",
    }
    response = await client.post(
        "/api/v1/vision/ocr",
        json=payload,
        headers=engineer_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_spans"] > 0
    assert "OMNIFORGE" in data["full_text"]
    assert len(data["spans"]) > 0


@pytest.mark.asyncio
async def test_video_tracking_endpoint(client: AsyncClient, engineer_headers: dict):
    payload = {
        "frames_base64": ["", "", ""],  # 3 frames
        "fps": 30.0,
        "confidence_threshold": 0.25,
    }
    response = await client.post(
        "/api/v1/vision/track/video",
        json=payload,
        headers=engineer_headers,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["total_frames_processed"] == 3
    assert data["total_unique_tracks"] > 0
    assert len(data["frame_results"]) == 3
