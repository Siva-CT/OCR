import cv2
import numpy as np
from typing import List
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

# Configurable Constants
YOLO_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_CONFIDENCE_THRESHOLD", "0.4"))
MIN_LABEL_WIDTH = int(os.environ.get("MIN_LABEL_WIDTH", "250"))
MIN_LABEL_HEIGHT = int(os.environ.get("MIN_LABEL_HEIGHT", "120"))

import os

# Load Model Configuration globally if available
detector_model = None
try:
    if YOLO is None:
        print("WARNING: ultralytics not installed. YOLO detection disabled.")
        detector_model = None
    else:
        custom_model_path = os.path.abspath("models/yolo-label-detector/weights/best.pt")
        if os.path.exists(custom_model_path):
            print(f"Loading CUSTOM YOLO SMT label detector from: {custom_model_path}")
            detector_model = YOLO(custom_model_path)
        else:
            print("Custom YOLO model not found. Falling back to generic yolov8n.pt...")
            detector_model = YOLO("yolov8n.pt")
except Exception as e:
    print(f"Failed to load YOLO models. Error: {e}")
    print("YOLO detection will be unavailable.")
    detector_model = None

def detect_labels_yolo(image: np.ndarray) -> List[np.ndarray]:
    """
    Attempts to rapidly isolate SMT label frames via YOLO Object Detection.
    Returns:
        List of cropped label images
    """
    if not YOLO or not detector_model:
        raise Exception("YOLO ultralytics not installed or Model missing.")

    # YOLO infers using standard BGR format natively. 
    results = detector_model(image, verbose=False)
    
    bounding_boxes = []
    
    # Process First (and only) image prediction output
    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            
            # Confidence Check
            if conf < YOLO_CONFIDENCE_THRESHOLD:
                continue
                
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w = x2 - x1
            h = y2 - y1
            
            # Dimension Threshold Validation (must be sufficiently large to be a label structure)
            if w >= MIN_LABEL_WIDTH and h >= MIN_LABEL_HEIGHT:
                bounding_boxes.append((x1, y1, w, h))
                
    if not bounding_boxes:
        return []

    # Sort descending top-to-bottom relative to image orientation so parsing stacks downward
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[1])

    cropped_labels = []
    # Mask and crop regions with padding
    for (x, y, w, h) in bounding_boxes:
        pad = 30
        img_h, img_w = image.shape[:2]
        
        px1 = max(0, x - pad)
        py1 = max(0, y - pad)
        px2 = min(img_w, x + w + pad)
        py2 = min(img_h, y + h + pad)

        crop = image[py1:py2, px1:px2]
        cropped_labels.append(crop)

    return cropped_labels
