"""
Minimal YOLO human detection module with default model path.

Usage:
    # Uses default MODEL_PATH unless overridden
    model = load_model()
    detections = detect_humans(model, frame)

Each detection is a dict: {"bbox": (x1,y1,x2,y2), "confidence": float}
"""
import numpy as np
from ultralytics import YOLO

# Default YOLO model weights path (provided by user)
MODEL_PATH = "/path/to/yolov5nu.pt"
# Default inference parameters
IMG_SIZE = 640
CONF_THRESHOLD = 0.3


def load_model(model_path: str = MODEL_PATH) -> YOLO:
    """Load and return a YOLO model from the provided weights path."""
    return YOLO(model_path)


def detect_humans(
    model: YOLO,
    frame: np.ndarray,
    imgsz: int = IMG_SIZE,
    conf_threshold: float = CONF_THRESHOLD,
) -> list:
    """
    Run YOLO inference on a BGR image to detect people.

    Args:
        model: Loaded YOLO instance
        frame: OpenCV BGR image array
        imgsz: Inference image size (pixels)
        conf_threshold: Minimum confidence for detections

    Returns:
        List of detection dicts: {"bbox": (x1, y1, x2, y2), "confidence": float}
    """
    results = model.predict(frame, imgsz=imgsz, conf=conf_threshold, verbose=False)
    detections = []
    # Iterate detected boxes, confidences, and class IDs
    for box, conf, cls in zip(
        results[0].boxes.xyxy,
        results[0].boxes.conf,
        results[0].boxes.cls,
    ):
        # Filter to person class (class ID 0)
        if int(cls) != 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        detections.append({
            "bbox": (x1, y1, x2, y2),
            "confidence": float(conf),
        })
    return detections