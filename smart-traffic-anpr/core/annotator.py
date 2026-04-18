"""
annotator.py — Draw boxes, labels, and line overlays on the video frame.
"""

import cv2
import numpy as np
from config import (
    LINE_COLOR_NORMAL, LINE_THICKNESS, BOX_THICKNESS, FONT_SCALE,
    DRAW_TRACK_ID, DRAW_CONFIDENCE, DRAW_PLATE_TEXT
)

class FrameAnnotator:
    def __init__(self, line_coords: tuple):
        self.line_start, self.line_end = line_coords

    def draw(self, frame: np.ndarray, tracked_vehicles: list[dict], crossing_events: list[dict], counts: dict) -> np.ndarray:
        """Annotates the given frame in-place (or creates a copy)."""
        annotated = frame.copy()
        
        # Draw virtual line
        cv2.line(annotated, self.line_start, self.line_end, LINE_COLOR_NORMAL, LINE_THICKNESS)
        
        # Highlight triggered events with a thick red circle
        event_ids = [e["track_id"] for e in crossing_events]
        if event_ids:
            cv2.line(annotated, self.line_start, self.line_end, (0, 0, 255), LINE_THICKNESS + 2)

        # Draw vehicle bounding boxes
        for v in tracked_vehicles:
            x1, y1, x2, y2 = v["bbox"]
            tid = v["track_id"]
            cls = v["class_name"]
            conf = v["confidence"]
            cx, cy = v["centroid"]
            
            # Change color if recently crossed
            color = (0, 0, 255) if tid in event_ids else (0, 255, 0)
            
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, BOX_THICKNESS)
            
            # Label
            label_parts = [cls]
            if DRAW_TRACK_ID:
                label_parts.append(f"#{tid}")
            if DRAW_CONFIDENCE:
                label_parts.append(f"{conf:.2f}")
            
            label = " ".join(label_parts)
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1)
            cv2.rectangle(annotated, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
            
            cv2.circle(annotated, (cx, cy), 4, color, -1)

        # Overlay counts
        y_offset = 30
        for cls, count in counts.items():
            text = f"{cls.capitalize()}: {count}"
            cv2.putText(annotated, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(annotated, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), 1, cv2.LINE_AA)
            y_offset += 30

        return annotated
