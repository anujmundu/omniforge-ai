"""
Unit tests for Multi-Object Spatial Tracker.
"""

from vision.base import BoundingBox, Detection
from vision.tracker import MultiObjectTracker


def test_multi_object_tracker_id_persistence_and_trajectory():
    tracker = MultiObjectTracker(max_age=5, min_hits=1, iou_threshold=0.2)

    # Frame 1: Person at (0.2, 0.2) -> (0.4, 0.6)
    det1 = [Detection(label="person", confidence=0.9, box=BoundingBox(xmin=0.2, ymin=0.2, xmax=0.4, ymax=0.6))]
    res1 = tracker.update(det1, frame_index=0, timestamp_ms=0.0)
    assert len(res1.active_tracks) == 1
    track_id = res1.active_tracks[0].track_id
    assert track_id == 1
    assert res1.active_tracks[0].age_frames == 1

    # Frame 2: Person slightly moved to (0.22, 0.21) -> (0.42, 0.61)
    det2 = [Detection(label="person", confidence=0.92, box=BoundingBox(xmin=0.22, ymin=0.21, xmax=0.42, ymax=0.61))]
    res2 = tracker.update(det2, frame_index=1, timestamp_ms=33.3)
    assert len(res2.active_tracks) == 1
    assert res2.active_tracks[0].track_id == track_id  # Persistent ID
    assert res2.active_tracks[0].age_frames == 2
    assert len(res2.active_tracks[0].history_centers) == 2


def test_multi_object_tracker_track_expiry():
    tracker = MultiObjectTracker(max_age=2, min_hits=1)

    det = [Detection(label="car", confidence=0.85, box=BoundingBox(xmin=0.1, ymin=0.1, xmax=0.3, ymax=0.3))]
    tracker.update(det, frame_index=0)
    assert len(tracker._tracks) == 1

    # Empty frames 1, 2, 3
    tracker.update([], frame_index=1)
    tracker.update([], frame_index=2)
    tracker.update([], frame_index=3)

    # Track should be pruned after exceeding max_age
    assert len(tracker._tracks) == 0
