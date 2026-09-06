"""
Multi-Object Spatial Tracking Engine with centroid and IoU association,
motion smoothing, and persistent identity assignment.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from vision.base import BaseTracker, BoundingBox, Detection, TrackedObject, TrackingResult


class MultiObjectTracker(BaseTracker):
    """
    Spatial Multi-Object Tracker (ByteTrack / Centroid IoU Association).
    Maintains persistent track IDs, trajectory history, and velocity vectors
    across sequential video frames.
    """

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 1,
        iou_threshold: float = 0.30,
        max_distance: float = 0.25,
    ) -> None:
        """
        Args:
            max_age: Maximum consecutive missing frames before track deletion.
            min_hits: Minimum detections before track is confirmed active.
            iou_threshold: Minimum IoU for spatial box association.
            max_distance: Maximum normalized Euclidean centroid distance for fallback matching.
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.max_distance = max_distance

        self._next_track_id: int = 1
        self._tracks: Dict[int, TrackedObject] = {}
        self._total_tracks_count: int = 0

    def reset(self) -> None:
        """Reset all tracking states."""
        self._next_track_id = 1
        self._tracks.clear()
        self._total_tracks_count = 0

    def update(
        self,
        detections: List[Detection],
        frame_index: int,
        timestamp_ms: Optional[float] = None,
    ) -> TrackingResult:
        """
        Associate frame detections with existing active tracks and spawn new tracks.
        """
        start_time = time.perf_counter()

        # Step 1: Predict / age existing tracks
        for track in self._tracks.values():
            track.time_since_update += 1

        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(self._tracks.keys())
        matched_pairs: List[Tuple[int, int]] = []  # (track_id, det_idx)

        # Step 2: IoU and Centroid association
        if unmatched_tracks and unmatched_dets:
            # Build cost matrix
            for track_id in unmatched_tracks:
                track = self._tracks[track_id]
                best_match_idx = -1
                best_score = -1.0

                for det_idx in unmatched_dets:
                    det = detections[det_idx]
                    
                    # Must share the same class label
                    if det.label != track.label:
                        continue

                    iou = track.current_box.iou(det.box)
                    
                    # Centroid distance
                    tc_x, tc_y = track.current_box.center
                    dc_x, dc_y = det.box.center
                    dist = np.sqrt((tc_x - dc_x) ** 2 + (tc_y - dc_y) ** 2)

                    # Combined score: IoU + proximity
                    if iou >= self.iou_threshold or dist <= self.max_distance:
                        score = iou + max(0.0, 1.0 - dist)
                        if score > best_score:
                            best_score = score
                            best_match_idx = det_idx

                if best_match_idx != -1 and best_match_idx in unmatched_dets:
                    matched_pairs.append((track_id, best_match_idx))
                    unmatched_dets.remove(best_match_idx)

        # Step 3: Update matched tracks
        for track_id, det_idx in matched_pairs:
            det = detections[det_idx]
            track = self._tracks[track_id]
            
            old_center = track.current_box.center
            new_center = det.box.center
            vx = new_center[0] - old_center[0]
            vy = new_center[1] - old_center[1]

            track.current_box = det.box
            track.confidence = det.confidence
            track.history_centers.append(new_center)
            if len(track.history_centers) > 50:
                track.history_centers.pop(0)

            track.age_frames += 1
            track.time_since_update = 0
            track.velocity = (round(vx, 4), round(vy, 4))

        # Step 4: Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            tid = self._next_track_id
            self._next_track_id += 1
            self._total_tracks_count += 1

            new_track = TrackedObject(
                track_id=tid,
                label=det.label,
                confidence=det.confidence,
                current_box=det.box,
                history_centers=[det.box.center],
                age_frames=1,
                time_since_update=0,
                velocity=(0.0, 0.0),
            )
            self._tracks[tid] = new_track

        # Step 5: Remove expired tracks
        dead_tracks = [
            tid for tid, track in self._tracks.items()
            if track.time_since_update > self.max_age
        ]
        for tid in dead_tracks:
            del self._tracks[tid]

        # Active confirmed tracks
        active = [
            track for track in self._tracks.values()
            if track.time_since_update == 0 and track.age_frames >= self.min_hits
        ]

        latency = (time.perf_counter() - start_time) * 1000.0

        return TrackingResult(
            frame_index=frame_index,
            timestamp_ms=timestamp_ms,
            active_tracks=active,
            total_tracks_observed=self._total_tracks_count,
            inference_latency_ms=round(latency, 2),
        )
