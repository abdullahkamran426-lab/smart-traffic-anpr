"""
streamlit_app.py — Smart Traffic ANPR System (Streamlit Cloud Optimized)
Main entry point for video upload, processing, and dashboard.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from datetime import datetime
from pathlib import Path
import tempfile
import logging
import os
import sys

# Setup logging for cloud
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Traffic ANPR System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Session State ────────────────────────────────────────────────────────────
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_file_path' not in st.session_state:
    st.session_state.uploaded_file_path = None

# ─── Cached Model Loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector():
    """Load and cache the vehicle detection model."""
    try:
        from core.detector import VehicleDetector
        detector = VehicleDetector()
        return detector
    except Exception as e:
        logger.error(f"Failed to load detector: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_tracker():
    """Load and cache the tracker."""
    try:
        from core.tracker import VehicleTracker
        tracker = VehicleTracker()
        return tracker
    except Exception as e:
        logger.error(f"Failed to load tracker: {e}")
        return None

@st.cache_resource(show_spinner=False)
def load_anpr():
    """Load and cache the ANPR engine."""
    try:
        from core.anpr import ANPREngine
        anpr = ANPREngine()
        return anpr
    except Exception as e:
        logger.error(f"Failed to load ANPR engine: {e}")
        return None

@st.cache_resource(show_spinner=False)
def get_database():
    """Get database manager."""
    try:
        from database.db_manager import DatabaseManager
        db = DatabaseManager()
        return db
    except Exception as e:
        logger.error(f"Failed to load database: {e}")
        return None

# ─── Video Processing Pipeline ────────────────────────────────────────────────
def process_video(video_path, progress_bar, status_text):
    """
    Process video and detect/track vehicles with license plates.
    Optimized for cloud with limited RAM.
    """
    from config import (
        RESIZE_WIDTH, RESIZE_HEIGHT, FRAME_SKIP,
        MAX_VIDEO_DURATION, MAX_FILE_SIZE_MB
    )
    from core.line_crossing import LineCrossingDetector
    from core.annotator import FrameAnnotator

    # Check file size
    file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise ValueError(f"File too large: {file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit")

    # Load models
    detector = load_detector()
    tracker = load_tracker()
    anpr = load_anpr()
    db = get_database()

    if detector is None:
        raise RuntimeError("Failed to load detection model")

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video file")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    # Check duration
    if duration > MAX_VIDEO_DURATION:
        cap.release()
        raise ValueError(f"Video too long: {duration:.0f}s > {MAX_VIDEO_DURATION}s limit")

    # Initialize components
    line_det = LineCrossingDetector(RESIZE_WIDTH, RESIZE_HEIGHT)
    annotator = FrameAnnotator(line_det.get_line_coords())

    # Initialize counters
    frame_idx = 0
    processed_plates = []
    vehicle_data = []

    # Output video path
    output_path = None

    status_text.text("Processing video... Please wait.")

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1

            # Update progress
            if frame_idx % 30 == 0:
                progress = min(frame_idx / total_frames, 1.0) if total_frames > 0 else 0
                progress_bar.progress(progress, f"Processing frame {frame_idx}/{total_frames}")

            # Skip frames for performance
            if frame_idx % FRAME_SKIP != 0:
                continue

            # Resize for faster processing
            frame = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT))

            # Detect vehicles
            detections = detector.detect(frame)

            # Track vehicles
            tracked = tracker.update(detections, frame)

            # Check line crossings
            events = line_det.update(tracked)

            # Process plates for crossing events
            for event in events:
                tid = event["track_id"]
                cls = event["class_name"]

                # Find vehicle's bounding box
                vehicle_data_dict = next(
                    (v for v in tracked if v["track_id"] == tid), None
                )
                if not vehicle_data_dict:
                    continue

                x1, y1, x2, y2 = vehicle_data_dict["bbox"]
                vehicle_crop = frame[y1:y2, x1:x2]

                if vehicle_crop.size == 0:
                    continue

                # Run ANPR
                plate_result = None
                if anpr is not None:
                    try:
                        plate_result = anpr.run(vehicle_crop)
                    except Exception as e:
                        logger.warning(f"ANPR failed for track {tid}: {e}")

                plate_text = plate_result["plate_text"] if plate_result else None
                ocr_conf = plate_result["ocr_confidence"] if plate_result else None

                # Store in database
                if db is not None:
                    try:
                        db.insert_vehicle(
                            vehicle_type=cls,
                            plate_number=plate_text,
                            ocr_confidence=ocr_conf,
                            video_source=Path(video_path).name
                        )
                    except Exception as e:
                        logger.warning(f"Failed to insert to DB: {e}")

                # Store for display
                if plate_text:
                    processed_plates.append({
                        "frame": frame_idx,
                        "vehicle_type": cls,
                        "plate_number": plate_text,
                        "confidence": ocr_conf,
                        "track_id": tid
                    })

                vehicle_data.append({
                    "frame": frame_idx,
                    "track_id": tid,
                    "vehicle_type": cls,
                    "plate": plate_text or "N/A"
                })

    finally:
        cap.release()

    counts = line_det.get_counts()

    return {
        "total_frames": frame_idx,
        "counts": counts,
        "plates": processed_plates,
        "vehicles": vehicle_data,
        "duration": duration
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚗 Smart Traffic ANPR")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["📤 Upload & Process", "📊 Dashboard", "📋 Plate Log"]
    )

    st.markdown("---")

    # Model status
    st.subheader("🤖 Model Status")
    detector = load_detector()
    anpr = load_anpr()
    db = get_database()

    if detector:
        st.success("✅ Vehicle Detector")
    else:
        st.error("❌ Vehicle Detector")

    if anpr:
        if anpr.plate_model_available:
            st.success("✅ Plate Detector + OCR")
        else:
            st.warning("⚠️ OCR Only (No Plate Model)")
    else:
        st.error("❌ ANPR Engine")

    if db:
        st.success("✅ Database")
    else:
        st.error("❌ Database")

    st.markdown("---")
    st.caption("v1.0.0 • Cloud Optimized")

# ─── Upload & Process Page ────────────────────────────────────────────────────
if page == "📤 Upload & Process":
    st.title("📤 Upload Traffic Video")
    st.markdown("Upload a video file to detect vehicles and read license plates.")

    # Info box
    st.info("""
    **Guidelines:**
    - Maximum file size: **50 MB**
    - Maximum duration: **60 seconds**
    - Supported formats: MP4, AVI, MOV, MKV
    - For best results, use clear footage with visible license plates
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a traffic video for ANPR processing"
    )

    if uploaded_file is not None:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
            st.session_state.uploaded_file_path = tmp_path

        # Show video preview
        st.subheader("🎬 Video Preview")
        st.video(tmp_path)

        # Process button
        col1, col2 = st.columns([1, 3])
        with col1:
            process_btn = st.button(
                "🚀 Process Video",
                type="primary",
                disabled=st.session_state.processing,
                use_container_width=True
            )

        with col2:
            if st.session_state.processing:
                st.info("⏳ Processing... Please wait. This may take a few minutes.")

        if process_btn and not st.session_state.processing:
            st.session_state.processing = True
            st.session_state.results = None
            st.rerun()

        if st.session_state.processing:
            # Create progress bar
            progress_bar = st.progress(0, "Initializing...")
            status_text = st.empty()

            try:
                # Process the video
                results = process_video(tmp_path, progress_bar, status_text)
                st.session_state.results = results

                progress_bar.empty()
                status_text.success("✅ Processing complete!")

            except Exception as e:
                progress_bar.empty()
                status_text.error(f"❌ Error: {str(e)}")
                logger.exception("Processing failed")

            finally:
                st.session_state.processing = False

            # Rerun to show results
            st.rerun()

        # Display results
        if st.session_state.results:
            results = st.session_state.results

            st.markdown("---")
            st.subheader("📊 Processing Results")

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Vehicles", results["counts"].get("total", 0))
            col2.metric("Cars", results["counts"].get("car", 0))
            col3.metric("Trucks", results["counts"].get("truck", 0))
            col4.metric("Plates Read", len(results["plates"]))

            # Plates detected
            if results["plates"]:
                st.subheader("📝 Detected Plates")
                plates_df = pd.DataFrame(results["plates"])
                plates_df = plates_df.rename(columns={
                    "vehicle_type": "Type",
                    "plate_number": "Plate",
                    "confidence": "Confidence",
                    "track_id": "Track ID"
                })
                st.dataframe(plates_df[["Plate", "Type", "Confidence", "Track ID"]], use_container_width=True)

                # Download button
                csv = plates_df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Results CSV",
                    data=csv,
                    file_name=f"anpr_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No license plates detected in this video.")

            # Cleanup temp file
            try:
                os.unlink(tmp_path)
            except:
                pass

# ─── Dashboard Page ───────────────────────────────────────────────────────────
elif page == "📊 Dashboard":
    st.title("📊 Traffic Analytics Dashboard")

    db = get_database()

    if db is None:
        st.error("❌ Database not available")
    else:
        try:
            # Summary stats
            stats = db.get_summary_stats()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Vehicles", stats["total_vehicles"])
            col2.metric("Plates Read", stats["plates_read"])
            col3.metric("OCR Success Rate", f"{stats['ocr_success_rate']}%")
            col4.metric("Avg OCR Confidence", f"{stats['avg_ocr_conf']:.2f}")

            st.divider()

            # Charts
            col_a, col_b = st.columns([1, 2])

            with col_a:
                st.subheader("🚗 Vehicle Types")
                counts = db.get_counts_by_class()
                if counts:
                    # Filter out 'total' key for chart
                    chart_counts = {k: v for k, v in counts.items() if k != 'total'}
                    count_df = pd.DataFrame(chart_counts.items(), columns=["Type", "Count"])
                    fig = px.pie(count_df, values="Count", names="Type")
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data yet.")

            with col_b:
                st.subheader("📈 Hourly Traffic Trend")
                trend = db.get_hourly_trend(hours=24)
                if trend:
                    trend_df = pd.DataFrame(trend)
                    fig = px.bar(
                        trend_df, x="hour", y="count",
                        labels={"hour": "Time", "count": "Vehicles"}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No trend data available.")

        except Exception as e:
            st.error(f"Error loading dashboard: {e}")

# ─── Plate Log Page ───────────────────────────────────────────────────────────
elif page == "📋 Plate Log":
    st.title("📋 License Plate Log")

    db = get_database()

    if db is None:
        st.error("❌ Database not available")
    else:
        # Search
        search_query = st.text_input("🔍 Search by plate number", placeholder="e.g., ABC123")

        try:
            if search_query:
                logs = db.search_by_plate(search_query)
                st.caption(f"Showing results for: '{search_query}'")
            else:
                logs = db.get_recent_logs(limit=100)

            if logs:
                df = pd.DataFrame(logs)
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.rename(columns={
                    "vehicle_type": "Type",
                    "plate_number": "Plate",
                    "ocr_confidence": "Confidence",
                    "timestamp": "Time",
                    "video_source": "Source"
                })

                st.dataframe(
                    df[["Time", "Type", "Plate", "Confidence", "Source"]],
                    use_container_width=True,
                    height=500
                )

                # Download
                csv = df.to_csv(index=False)
                st.download_button(
                    "⬇️ Download Full Log",
                    data=csv,
                    file_name=f"plate_log_{datetime.now().date()}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No records found. Upload and process a video to see results.")

        except Exception as e:
            st.error(f"Error loading log: {e}")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Smart Traffic ANPR System | Built with Streamlit, YOLOv8, and EasyOCR")
