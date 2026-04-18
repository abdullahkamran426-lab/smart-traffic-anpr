"""
anpr.py — Two-stage ANPR pipeline.
Stage 1: YOLO-based license plate region detection.
Stage 2: OCR extraction using EasyOCR or Tesseract.
Only returns results above the configured confidence threshold.
"""

import re
import logging
import numpy as np
import cv2
import easyocr
from ultralytics import YOLO
from config import (
    PLATE_MODEL_PATH, PLATE_CONF_THRESHOLD, OCR_CONF_THRESHOLD,
    OCR_ENGINE, OCR_LANGUAGES, OCR_GPU,
    PLATE_TEXT_MIN_LENGTH, PLATE_TEXT_MAX_LENGTH, PLATE_CHAR_WHITELIST
)

logger = logging.getLogger(__name__)


class ANPREngine:
    """
    Detects license plates in a vehicle crop and extracts the plate text.
    Falls back to full-frame OCR if plate model not available.
    """

    def __init__(self):
        self.plate_model = None
        self.reader = None
        self.plate_model_available = False

        # Try to load plate detection model
        if PLATE_MODEL_PATH.exists():
            try:
                logger.info("Loading license plate detection model...")
                self.plate_model = YOLO(str(PLATE_MODEL_PATH))
                self.plate_model_available = True
                logger.info("Plate detection model ready.")
            except Exception as e:
                logger.warning(f"Could not load plate model: {e}")
                logger.warning("Will use full-frame OCR as fallback.")
        else:
            logger.warning(f"Plate model not found at {PLATE_MODEL_PATH}")
            logger.warning("Download a license plate model or use full-frame OCR.")

        # Initialize EasyOCR (CPU only)
        try:
            logger.info("Initializing EasyOCR reader (CPU mode)...")
            self.reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
            logger.info("EasyOCR ready.")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            self.reader = None

    def _run_full_frame_ocr(self, vehicle_crop: np.ndarray) -> dict | None:
        """Fallback: Run OCR on full vehicle crop without plate detection."""
        if self.reader is None:
            return None

        processed = self._preprocess_plate(vehicle_crop)
        ocr_results = self.reader.readtext(processed, allowlist=PLATE_CHAR_WHITELIST)

        if not ocr_results:
            return None

        best_result = max(ocr_results, key=lambda r: r[2])
        raw_text = best_result[1]
        confidence = float(best_result[2])
        plate_text = self._normalize_plate_text(raw_text)

        if not self._is_valid_plate(plate_text):
            return None

        if confidence < OCR_CONF_THRESHOLD:
            return None

        return {
            "plate_text": plate_text,
            "ocr_confidence": round(confidence, 3),
            "plate_bbox": [0, 0, vehicle_crop.shape[1], vehicle_crop.shape[0]]
        }

    def _preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """
        Apply image preprocessing to improve OCR accuracy on the plate crop.
        Steps: resize → grayscale → CLAHE → sharpen → threshold
        """
        # Resize to standard height
        h, w = plate_img.shape[:2]
        scale = 64 / h
        plate_img = cv2.resize(plate_img, (int(w * scale), 64))

        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)

        # CLAHE for contrast normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Sharpening kernel
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return thresh

    def _normalize_plate_text(self, text: str) -> str:
        """
        Clean OCR output: uppercase, remove non-alphanumeric, strip whitespace.
        """
        text = text.upper().strip()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def _is_valid_plate(self, text: str) -> bool:
        """Basic validation on extracted plate text."""
        return PLATE_TEXT_MIN_LENGTH <= len(text) <= PLATE_TEXT_MAX_LENGTH

    def run(self, vehicle_crop: np.ndarray) -> dict | None:
        """
        Run full ANPR on a cropped vehicle image.
        Falls back to full-frame OCR if plate model not available.
        """
        # Check if EasyOCR is available
        if self.reader is None:
            logger.error("OCR engine not available")
            return None

        # Use plate detection if available, otherwise fallback to full-frame OCR
        if self.plate_model_available and self.plate_model is not None:
            return self._run_with_plate_detection(vehicle_crop)
        else:
            return self._run_full_frame_ocr(vehicle_crop)

    def _run_with_plate_detection(self, vehicle_crop: np.ndarray) -> dict | None:
        """Run ANPR with plate detection model."""
        # Stage 1: Detect plate bounding box
        try:
            results = self.plate_model.predict(
                source=vehicle_crop,
                conf=PLATE_CONF_THRESHOLD,
                verbose=False
            )[0]
        except Exception as e:
            logger.error(f"Plate detection failed: {e}")
            return self._run_full_frame_ocr(vehicle_crop)

        if len(results.boxes) == 0:
            logger.debug("No license plate detected in vehicle crop.")
            return None

        # Take highest-confidence plate detection
        best_box = max(results.boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

        # Clamp coordinates
        h_crop, w_crop = vehicle_crop.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w_crop, x2), min(h_crop, y2)

        plate_crop = vehicle_crop[y1:y2, x1:x2]
        if plate_crop.size == 0:
            return None

        # Stage 2: Preprocess + OCR
        processed = self._preprocess_plate(plate_crop)

        ocr_results = self.reader.readtext(processed, allowlist=PLATE_CHAR_WHITELIST)
        if not ocr_results:
            return None

        best_result = max(ocr_results, key=lambda r: r[2])
        raw_text = best_result[1]
        confidence = float(best_result[2])

        plate_text = self._normalize_plate_text(raw_text)

        if not self._is_valid_plate(plate_text):
            logger.debug(f"Invalid plate text discarded: '{plate_text}'")
            return None

        if confidence < OCR_CONF_THRESHOLD:
            logger.debug(f"OCR confidence too low: {confidence:.2f} < {OCR_CONF_THRESHOLD}")
            return None

        logger.info(f"Plate detected: '{plate_text}' (conf={confidence:.2f})")

        return {
            "plate_text": plate_text,
            "ocr_confidence": round(confidence, 3),
            "plate_bbox": [x1, y1, x2, y2]
        }
