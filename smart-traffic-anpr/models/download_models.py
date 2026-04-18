import os
from pathlib import Path
from ultralytics import YOLO

def main():
    model_dir = Path(__file__).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading YOLOv8n for vehicle detection...")
    model_path = model_dir / "yolov8n.pt"
    # Loading the model will automatically download the weights if they don't exist
    YOLO(str(model_path))
    print(f"✓ Vehicle detection model ready: {model_path}")
    
    plate_model_path = model_dir / "license_plate_detector.pt"
    if not plate_model_path.exists():
        print("\n⚠️  WARNING: License plate model not found!")
        print(f"   Expected at: {plate_model_path}")
        print("\n   You need to either:")
        print("   1. Download a pre-trained license plate detection model from:")
        print("      https://github.com/ultralytics/assets or Roboflow Universe")
        print("   2. Train your own using YOLOv8 on license plate dataset")
        print("   3. For testing/demo, the system will skip plate detection")
        print("\n   Place the model file at the expected path and restart.")
    else:
        print(f"✓ License plate model ready: {plate_model_path}")
    
    print("\nDone.")

if __name__ == "__main__":
    main()
