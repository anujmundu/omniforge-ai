"""
High-throughput asynchronous video frame stream ingestion and processing pipeline.
"""

from __future__ import annotations

import asyncio
from typing import AsyncGenerator, List, Optional, Tuple

import numpy as np

from vision.base import BaseDetector, BaseTracker, DetectionResult, TrackingResult


class VideoStreamProcessor:
    """
    Asynchronous Video Stream Ingestion and Tracking Pipeline.
    Buffers incoming video frames, applies temporal stride sampling,
    and runs real-time detection & tracking in non-blocking async loops.
    """

    def __init__(
        self,
        detector: BaseDetector,
        tracker: BaseTracker,
        stride: int = 1,
        max_buffer_size: int = 100,
    ) -> None:
        """
        Args:
            detector: Object detection engine instance.
            tracker: Multi-object spatial tracking engine instance.
            stride: Process every N-th frame (stride=1 processes all frames).
            max_buffer_size: Maximum queue capacity for memory safety.
        """
        self.detector = detector
        self.tracker = tracker
        self.stride = max(1, stride)
        self.max_buffer_size = max_buffer_size
        self._queue: asyncio.Queue[Tuple[int, float, np.ndarray]] = asyncio.Queue(maxsize=max_buffer_size)
        self._is_running = False

    async def ingest_frame(self, frame_index: int, timestamp_ms: float, frame: np.ndarray) -> bool:
        """
        Non-blocking ingestion of a single video frame into the processing queue.
        Returns False if buffer is full (backpressure signal).
        """
        try:
            self._queue.put_nowait((frame_index, timestamp_ms, frame))
            return True
        except asyncio.QueueFull:
            return False

    def start(self) -> None:
        """Explicitly mark processor as active."""
        self._is_running = True

    async def process_stream(
        self,
        confidence_threshold: float = 0.25,
        classes: Optional[List[str]] = None,
    ) -> AsyncGenerator[Tuple[DetectionResult, TrackingResult], None]:
        """
        Asynchronously consume and yield detection & tracking results for buffered frames.
        """
        while True:
            try:
                frame_idx, ts_ms, frame = await asyncio.wait_for(self._queue.get(), timeout=0.05)
            except asyncio.TimeoutError:
                if not self._is_running:
                    break
                continue

            # Apply frame stride sampling
            if frame_idx % self.stride != 0:
                self._queue.task_done()
                continue

            # 1. Run detection
            det_result = self.detector.detect(
                image=frame,
                confidence_threshold=confidence_threshold,
                classes=classes,
                frame_index=frame_idx,
                timestamp_ms=ts_ms,
            )

            # 2. Update tracker
            track_result = self.tracker.update(
                detections=det_result.detections,
                frame_index=frame_idx,
                timestamp_ms=ts_ms,
            )

            self._queue.task_done()
            yield (det_result, track_result)

    def stop(self) -> None:
        """Signal the stream processor to stop receiving new frames."""
        self._is_running = False

    def process_frame_batch(
        self,
        frames: List[np.ndarray],
        fps: float = 30.0,
        confidence_threshold: float = 0.25,
        classes: Optional[List[str]] = None,
    ) -> List[Tuple[DetectionResult, TrackingResult]]:
        """
        Synchronous batch processing helper for pre-recorded video frame arrays.
        """
        results: List[Tuple[DetectionResult, TrackingResult]] = []
        interval_ms = 1000.0 / fps

        for i, frame in enumerate(frames):
            if i % self.stride != 0:
                continue

            ts_ms = round(i * interval_ms, 2)
            det_res = self.detector.detect(
                image=frame,
                confidence_threshold=confidence_threshold,
                classes=classes,
                frame_index=i,
                timestamp_ms=ts_ms,
            )
            track_res = self.tracker.update(
                detections=det_res.detections,
                frame_index=i,
                timestamp_ms=ts_ms,
            )
            results.append((det_res, track_res))

        return results
