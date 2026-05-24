# Smart Traffic ANPR System

AI-powered vehicle detection and license plate recognition system optimized for **Streamlit Cloud** deployment.

## Features

- 🎥 **Video Upload**: Upload traffic videos (MP4, AVI, MOV, MKV)
- 🚗 **Vehicle Detection**: YOLOv8-based vehicle detection (cars, trucks, buses, motorcycles)
- 🔤 **License Plate Recognition**: EasyOCR-based plate reading
- 📊 **Analytics Dashboard**: Real-time traffic statistics and trends
- 🔍 **Plate Search**: Search detected plates by partial match
- ⬇️ **CSV Export**: Download results for further analysis

## Streamlit Cloud Deployment

### Step 1: Fork/Clone Repository

```bash
git clone https://github.com/yourusername/smart-traffic-anpr.git
cd smart-traffic-anpr
```

### Step 2: Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set **Main file path** to: `streamlit_app.py`
6. Click "Deploy"

### Step 3: Configuration (Optional)

Create a `.streamlit/config.toml` file for custom settings:

```toml
[server]
maxUploadSize = 50

[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#262730"
textColor = "#FAFAFA"
```

## Local Development

### Prerequisites

- Python 3.9+
- 4GB+ RAM
- (Optional) CUDA GPU for faster processing

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/smart-traffic-anpr.git
cd smart-traffic-anpr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

### Optional: Download License Plate Model

For better accuracy, download a specialized license plate detection model:

```bash
# Place model at models/license_plate_detector.pt
# Download from Roboflow Universe or train your own with YOLOv8
```

Without a plate model, the system will use full-frame OCR (less accurate but functional).

## Project Structure

```
smart-traffic-anpr/
├── streamlit_app.py          # Main entry point (Streamlit UI)
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── core/
│   ├── detector.py          # YOLOv8 vehicle detection
│   ├── tracker.py           # Simple IoU tracker
│   ├── anpr.py              # License plate OCR
│   ├── line_crossing.py     # Virtual line detection
│   └── annotator.py         # Frame visualization
├── database/
│   └── db_manager.py        # SQLite database operations
├── models/
│   └── download_models.py   # Model download utilities
└── data/
    └── (uploaded videos)
```

## Cloud Optimization Notes

### Changes Made for Cloud Deployment

1. **Replaced `boxmot` with Simple IoU Tracker**: Removes heavy dependencies
2. **CPU-only PyTorch**: Uses `torch==2.3.0+cpu` for cloud compatibility
3. **OpenCV Headless**: Uses `opencv-python-headless` for server environments
4. **Model Caching**: `@st.cache_resource` prevents reloading on every interaction
5. **File Size Limits**: 50MB max upload, 60s max duration
6. **Removed FastAPI**: Not needed for Streamlit Cloud (single-process)
7. **Temp Directory Handling**: Uses `/tmp` on Linux, local temp on Windows
8. **Graceful Degradation**: Works without plate detection model (OCR-only mode)

### Performance Tips

- **Frame Skip**: Set to 3 (process every 3rd frame)
- **Resolution**: Downscaled to 640x360 for faster processing
- **Batch Processing**: Videos processed frame-by-frame to manage memory
- **Progress Bar**: Shows real-time processing status

## Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'cv2'`
- **Fix**: Ensure `opencv-python-headless` is in requirements.txt

**Issue**: EasyOCR fails to load
- **Fix**: First run downloads models (~100MB), be patient

**Issue**: Out of memory on large videos
- **Fix**: Reduce `MAX_VIDEO_DURATION` in config.py or upload shorter clips

**Issue**: No plates detected
- **Fix**: System works without plate model; for better accuracy, add a trained YOLO plate detector

## License

MIT License - See LICENSE file for details

## Credits

- YOLOv8 by Ultralytics
- EasyOCR by Jaided AI
- Streamlit for the web framework
