"""
config.py — All system-wide settings.
Change values here to tune the pipeline without touching any other file.
"""

import os
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR              = Path(__file__).parent
MODEL_DIR             = BASE_DIR / "models"
DATA_DIR              = BASE_DIR / "data"
OUTPUT_DIR            = BASE_DIR / "output"
DATABASE_DIR          = BASE_DIR / "database"
LOG_DIR               = BASE_DIR / "output" / "logs"
TEMP_DIR              = Path("/tmp") if os.name != "nt" else (BASE_DIR / "temp")

# Ensure directories exist
for d in [MODEL_DIR, DATA_DIR, OUTPUT_DIR, DATABASE_DIR, LOG_DIR, TEMP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

VEHICLE_MODEL_PATH    = MODEL_DIR / "yolov8n.pt"
PLATE_MODEL_PATH      = MODEL_DIR / "license_plate_detector.pt"
DATABASE_PATH         = DATABASE_DIR / "traffic.db"
LOG_FILE              = LOG_DIR / "pipeline.log"

# Model URLs for auto-download
YOLOV8N_URL           = "https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"

# ─── Video Input ──────────────────────────────────────────────────────────────
VIDEO_SOURCE          = str(DATA_DIR / "sample_traffic_1.mp4")
# To use webcam: VIDEO_SOURCE = 0
# To use RTSP stream: VIDEO_SOURCE = "rtsp://192.168.1.100/stream"
FRAME_SKIP            = 3          # Process every Nth frame (higher = faster, less accurate)
RESIZE_WIDTH          = 640        # Optimized for cloud (was 1280)
RESIZE_HEIGHT         = 360        # Optimized for cloud (was 720)
MAX_VIDEO_DURATION    = 60         # Max video length in seconds for cloud
MAX_FILE_SIZE_MB      = 50         # Max upload file size

# ─── Virtual Line ─────────────────────────────────────────────────────────────
# Coordinates as fractions of frame size (0.0 to 1.0)
# Default: horizontal line at 60% down the frame
LINE_START_RATIO      = (0.0, 0.6)   # (x_ratio, y_ratio) for line start
LINE_END_RATIO        = (1.0, 0.6)   # (x_ratio, y_ratio) for line end
LINE_THICKNESS        = 3
LINE_COLOR_NORMAL     = (0, 255, 255)  # Yellow (BGR)
LINE_COLOR_TRIGGERED  = (0, 0, 255)    # Red (BGR)

# ─── Detection Thresholds ─────────────────────────────────────────────────────
VEHICLE_CONF_THRESHOLD  = 0.45    # Min confidence for vehicle detection
PLATE_CONF_THRESHOLD    = 0.50    # Min confidence for plate detection
OCR_CONF_THRESHOLD      = 0.75    # Min OCR confidence to store in DB
NMS_IOU_THRESHOLD       = 0.45    # Non-max suppression overlap threshold

# ─── Vehicle Classes (COCO dataset class IDs) ─────────────────────────────────
VEHICLE_CLASSES = {
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck"
}

# ─── Tracking ─────────────────────────────────────────────────────────────────
TRACKER_TYPE          = "bytetrack"   # "bytetrack" or "deepsort"
MAX_DISAPPEARED       = 30            # Frames before track is dropped
MIN_TRACK_LENGTH      = 5            # Min frames to consider a valid track
REENTRY_COOLDOWN      = 60           # Frames before same ID can cross again

# ─── OCR ──────────────────────────────────────────────────────────────────────
OCR_ENGINE            = "easyocr"    # Only easyocr supported in cloud
OCR_LANGUAGES         = ["en"]
OCR_GPU               = False        # Always False for cloud (no GPU)
PLATE_TEXT_MIN_LENGTH = 3            # Discard plates shorter than this
PLATE_TEXT_MAX_LENGTH = 12           # Discard plates longer than this
PLATE_CHAR_WHITELIST  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ─── Database ─────────────────────────────────────────────────────────────────
DB_ENGINE             = "sqlite"     # "sqlite" or "postgresql"
POSTGRES_URL          = os.getenv("POSTGRES_URL", "")  # Set in .env

# ─── Dashboard ────────────────────────────────────────────────────────────────
DASHBOARD_REFRESH_SEC = 3            # Auto-refresh interval (seconds)
CHART_HISTORY_MINUTES = 60           # Show last N minutes in trend chart
MAX_LOG_TABLE_ROWS    = 200          # Max rows shown in plate log table

# ─── Annotator ────────────────────────────────────────────────────────────────
DRAW_TRACK_ID         = True
DRAW_CONFIDENCE       = True
DRAW_PLATE_TEXT       = True
BOX_THICKNESS         = 2
FONT_SCALE            = 0.6

# ─── Logging ──────────────────────────────────────────────────────────────────
LOG_LEVEL             = "INFO"       # DEBUG / INFO / WARNING / ERROR
LOG_TO_FILE           = False        # Disabled for cloud (use console only)
LOG_TO_CONSOLE        = True
