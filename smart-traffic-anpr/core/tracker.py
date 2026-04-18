"""
tracker.py — Simple IoU-based tracker for cloud deployment.
No external dependencies - uses simple intersection-over-union matching.
"""

import logging
from config import VEHICLE_CLASSES

logger = logging.getLogger(__name__)


def compute_iou(box1, box2):
    """Compute Intersection over Union of two boxes."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    xi1 = max(x1_1, x1_2)
    yi1 = max(y1_1, y1_2)
    xi2 = min(x2_1, x2_2)
    yi2 = min(y2_1, y2_2)

    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


class VehicleTracker:
    """Simple IoU-based tracker - cloud optimized, no heavy dependencies."""

    def __init__(self, iou_threshold=0.3, max_disappeared=5):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.next_id = 1
        self.tracks = {}  # track_id -> {bbox, class_id, class_name, confidence, centroid, disappeared}
        self.track_history = {}  # track_id -> list of centroids
        logger.info("Initialized simple IoU tracker")

    def update(self, detections: list[dict], frame=None) -> list[dict]:
        """Update tracker with new detections."""
        if not detections:
            # Mark all tracks as disappeared
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.track_history[tid]
            return []

        # Simple greedy matching by IoU
        matched_track_ids = set()
        matched_det_indices = set()
        matches = []

        # Sort detections by confidence (highest first)
        sorted_dets = sorted(enumerate(detections), key=lambda x: x[1]["confidence"], reverse=True)

        for det_idx, det in sorted_dets:
            if det_idx in matched_det_indices:
                continue

            best_iou = self.iou_threshold
            best_track_id = None

            for tid, track in self.tracks.items():
                if tid in matched_track_ids:
                    continue

                iou = compute_iou(track["bbox"], det["bbox"])
                if iou > best_iou and track["class_id"] == det["class_id"]:
                    best_iou = iou
                    best_track_id = tid

            if best_track_id is not None:
                matches.append((best_track_id, det_idx))
                matched_track_ids.add(best_track_id)
                matched_det_indices.add(det_idx)

        # Update matched tracks
        for tid, det_idx in matches:
            det = detections[det_idx]
            cx, cy = (det["bbox"][0] + det["bbox"][2]) // 2, (det["bbox"][1] + det["bbox"][3]) // 2
            self.tracks[tid] = {
                "bbox": det["bbox"],
                "class_id": det["class_id"],
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "centroid": (cx, cy),
                "disappeared": 0
            }
            if tid not in self.track_history:
                self.track_history[tid] = []
            self.track_history[tid].append((cx, cy))

        # Mark unmatched tracks as disappeared
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]["disappeared"] += 1
                if self.tracks[tid]["disappeared"] > self.max_disappeared:
                    del self.tracks[tid]
                    del self.track_history[tid]

        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_det_indices:
                cx, cy = (det["bbox"][0] + det["bbox"][2]) // 2, (det["bbox"][1] + det["bbox"][3]) // 2
                tid = self.next_id
                self.next_id += 1
                self.tracks[tid] = {
                    "bbox": det["bbox"],
                    "class_id": det["class_id"],
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "centroid": (cx, cy),
                    "disappeared": 0
                }
                self.track_history[tid] = [(cx, cy)]

        # Build results
        results = []
        for tid, track in self.tracks.items():
            results.append({
                "track_id": tid,
                "bbox": track["bbox"],
                "class_id": track["class_id"],
                "class_name": track["class_name"],
                "confidence": round(track["confidence"], 3),
                "centroid": track["centroid"]
            })

        logger.debug(f"Active tracks: {len(results)}")
        return results

    def get_trail(self, track_id: int, max_points: int = 20) -> list[tuple]:
        """Return last N centroid positions."""
        history = self.track_history.get(track_id, [])
        return history[-max_points:]

    def reset(self):
        """Reset all tracks."""
        self.next_id = 1
        self.tracks = {}
        self.track_history = {}
