"""
line_crossing.py — Defines a virtual line on the road frame.
Detects when a vehicle's centroid crosses this line.
Maintains per-class vehicle counts.
Prevents re-counting via cooldown mechanism.
"""

import logging
from collections import defaultdict
from config import LINE_START_RATIO, LINE_END_RATIO, REENTRY_COOLDOWN

logger = logging.getLogger(__name__)


class LineCrossingDetector:
    """
    Defines a virtual counting line and detects crossing events.
    Uses centroid position change between frames to determine direction.
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.line_start = (
            int(LINE_START_RATIO[0] * frame_width),
            int(LINE_START_RATIO[1] * frame_height)
        )
        self.line_end = (
            int(LINE_END_RATIO[0] * frame_width),
            int(LINE_END_RATIO[1] * frame_height)
        )

        self.counts: dict[str, int] = defaultdict(int)
        self.crossed_ids: dict[int, int] = {}    # track_id -> last_crossed_frame
        self.prev_centroids: dict[int, tuple] = {}
        self.current_frame = 0

        logger.info(
            f"Line defined: {self.line_start} → {self.line_end}"
        )

    def _side_of_line(self, point: tuple) -> float:
        """
        Returns positive if point is below/right of line, negative if above/left.
        Uses cross-product of line vector and point vector.
        """
        x, y   = point
        x1, y1 = self.line_start
        x2, y2 = self.line_end
        return (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)

    def update(self, tracked_vehicles: list[dict]) -> list[dict]:
        """
        Check each tracked vehicle for a line crossing event.

        Args:
            tracked_vehicles: Output from VehicleTracker.update()

        Returns:
            List of crossing events this frame:
            [
              {
                "track_id": int,
                "class_name": str,
                "centroid": (cx, cy)
              },
              ...
            ]
        """
        self.current_frame += 1
        crossing_events = []

        for vehicle in tracked_vehicles:
            tid      = vehicle["track_id"]
            centroid = vehicle["centroid"]
            cls      = vehicle["class_name"]

            if tid in self.prev_centroids:
                prev = self.prev_centroids[tid]
                side_prev = self._side_of_line(prev)
                side_curr = self._side_of_line(centroid)

                crossed_recently = (
                    tid in self.crossed_ids and
                    self.current_frame - self.crossed_ids[tid] < REENTRY_COOLDOWN
                )

                if side_prev * side_curr < 0 and not crossed_recently:
                    # Sign change = crossed the line
                    self.counts[cls]         += 1
                    self.counts["total"]     += 1
                    self.crossed_ids[tid]     = self.current_frame

                    logger.info(
                        f"Line crossed! Class={cls}, TrackID={tid}, "
                        f"Total {cls}s: {self.counts[cls]}"
                    )

                    crossing_events.append({
                        "track_id":   tid,
                        "class_name": cls,
                        "centroid":   centroid
                    })

            self.prev_centroids[tid] = centroid

        return crossing_events

    def get_counts(self) -> dict:
        """Return current vehicle counts by class + total."""
        return dict(self.counts)

    def get_line_coords(self) -> tuple:
        """Return line start and end pixel coordinates."""
        return self.line_start, self.line_end

    def reset_counts(self):
        """Reset all counters (e.g., on new video session)."""
        self.counts.clear()
        self.crossed_ids.clear()
        self.prev_centroids.clear()
        logger.info("Line crossing counts reset.")
