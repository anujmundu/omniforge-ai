"""
Unit tests for VideoStreamProcessor.
"""

import asyncio
import numpy as np
import pytest
from vision.detector import ObjectDetector
from vision.stream import VideoStreamProcessor
from vision.tracker import MultiObjectTracker


@pytest.mark.asyncio
async def test_video_stream_processor_async_pipeline():
    detector = ObjectDetector()
    tracker = MultiObjectTracker()
    processor = VideoStreamProcessor(detector=detector, tracker=tracker, stride=1)

    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
    frame2 = np.zeros((480, 640, 3), dtype=np.uint8)

    await processor.ingest_frame(0, 0.0, frame1)
    await processor.ingest_frame(1, 33.3, frame2)
    processor.stop()

    results = []
    async for det, track in processor.process_stream():
        results.append((det, track))

    assert len(results) == 2
    assert results[0][0].frame_index == 0
    assert results[1][0].frame_index == 1


def test_video_stream_processor_batch():
    detector = ObjectDetector()
    tracker = MultiObjectTracker()
    processor = VideoStreamProcessor(detector=detector, tracker=tracker, stride=2)

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(5)]
    batch_res = processor.process_frame_batch(frames, fps=10.0)

    # Stride 2 over 5 frames (0, 2, 4) -> 3 processed frames
    assert len(batch_res) == 3
    assert batch_res[0][0].frame_index == 0
    assert batch_res[1][0].frame_index == 2
    assert batch_res[2][0].frame_index == 4
