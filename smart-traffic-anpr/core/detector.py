"""
detector.py — Wraps YOLOv8 model for vehicle detection.
Returns bounding boxes, class labels, and confidence scores.
"""

import logging
import os
import urllib.request
import numpy as np
from ultralytics import YOLO
from config import (
    VEHICLE_MODEL_PATH, VEHICLE_CONF_THRESHOLD,
    NMS_IOU_THRESHOLD, VEHICLE_CLASSES, YOLOV8N_URL
)

logger = logging.getLogger(__name__)


def download_model(model_path, url):
    """Download model if not exists."""
    if not model_path.exists():
        logger.info(f"Downloading model from {url}...")
        try:
            urllib.request.urlretrieve(url, model_path)
            logger.info(f"Model downloaded to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            # Fallback: let YOLO download it
            YOLO("yolov8n.pt")


class VehicleDetector:
    """
    Loads a YOLOv8 model and detects vehicles in a single frame.
    Filters results to only vehicle-class objects.
    """

    def __init__(self):
        logger.info(f"Loading vehicle detection model: {VEHICLE_MODEL_PATH}")
        # Auto-download if needed
        download_model(VEHICLE_MODEL_PATH, YOLOV8N_URL)
        try:
            self.model = YOLO(str(VEHICLE_MODEL_PATH))
            self.class_ids = list(VEHICLE_CLASSES.keys())
            logger.info("Vehicle detector ready.")
        except Exception as e:
            logger.error(f"Failed to load vehicle model: {e}")
            # Fallback to downloading via ultralytics
            self.model = YOLO("yolov8n.pt")
            self.class_ids = list(VEHICLE_CLASSES.keys())
            logger.info("Vehicle detector ready (fallback).")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        Run inference on a single BGR frame.

        Args:
            frame: OpenCV BGR image (numpy array)

        Returns:
            List of dicts:
            [
              {
                "bbox": [x1, y1, x2, y2],   # int pixel coords
                "class_id": int,
                "class_name": str,           # "car", "truck", etc.
                "confidence": float
              },
              ...
            ]
        """
        results = self.model.predict(
            source=frame,
            conf=VEHICLE_CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            classes=self.class_ids,
            verbose=False
        )[0]

        detections = []
        for box in results.boxes:
            class_id   = int(box.cls[0])
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append({
                "bbox":       [x1, y1, x2, y2],
                "class_id":   class_id,
                "class_name": VEHICLE_CLASSES.get(class_id, "vehicle"),
                "confidence": round(confidence, 3)
            })

        logger.debug(f"Detected {len(detections)} vehicles in frame.")
        return detections
