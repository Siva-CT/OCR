import os
import asyncio

import cv2
import numpy as np
from typing import Any, Dict, List, Tuple

from ocr_manager import run_ocr

MAX_WIDTH = 1280
LOGS_DIR = "/app/logs"
DEBUG_DIR = os.path.join(LOGS_DIR, "debug")

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

FIELD_CLASSES = {
    "part_number",
    "quantity",
    "capacitor",
    "voltage",
    "lot",
    "date_code",
    "barcode",
}

YOLO_FIELD_MODEL_PATH = os.environ.get(
    "YOLO_FIELD_MODEL_PATH",
    "models/yolo-field-detector/weights/best.pt",
)
YOLO_FIELD_CONFIDENCE_THRESHOLD = float(os.environ.get("YOLO_FIELD_CONFIDENCE_THRESHOLD", "0.35"))
YOLO_FIELD_IOU_THRESHOLD = float(os.environ.get("YOLO_FIELD_IOU_THRESHOLD", "0.5"))

_FIELD_DETECTOR = None


def resize_max_width(img, max_w=MAX_WIDTH):
    h, w = img.shape[:2]
    if w <= max_w:
        return img
    scale = max_w / float(w)
    new_size = (int(w * scale), int(h * scale))
    return cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)


def clahe_gray(img_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(img_gray)


def _order_box_points(points):
    pts = np.array(points, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    return np.array([
        pts[np.argmin(s)],
        pts[np.argmin(diff)],
        pts[np.argmax(s)],
        pts[np.argmax(diff)],
    ], dtype="float32")


def _estimate_skew_angle(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(thresh)
    if coords is None or len(coords) < 10:
        return 0.0

    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    return -angle


def deskew(img):
    angle = _estimate_skew_angle(img)
    if abs(angle) < 0.25:
        return img

    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def detect_and_crop_label(img):
    """
    Detect the label contour, perspective-warp it, then deskew the crop.
    Falls back to the full frame when no likely label contour is found.
    """
    resized = resize_max_width(img)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = clahe_gray(gray)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cropped = resized
    label_found = 0
    image_area = resized.shape[0] * resized.shape[1]

    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(contour)
        if area < image_area * 0.05:
            continue

        rect = cv2.minAreaRect(contour)
        width, height = rect[1]
        if width < 80 or height < 40:
            continue

        box = cv2.boxPoints(rect)
        ordered = _order_box_points(box)

        target_w = max(int(width), int(height))
        target_h = min(int(width), int(height))
        if target_w == 0 or target_h == 0:
            continue

        destination = np.array(
            [[0, 0], [target_w - 1, 0], [target_w - 1, target_h - 1], [0, target_h - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(ordered, destination)
        warped = cv2.warpPerspective(resized, matrix, (target_w, target_h))
        if warped.size == 0:
            continue

        cropped = warped
        label_found = 1
        break

    cropped = deskew(cropped)
    os.makedirs(LOGS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(LOGS_DIR, "cropped_label.jpg"), cropped)
    return cropped, label_found


def preprocess_for_ocr(img):
    resized = resize_max_width(img)
    straightened = deskew(resized)
    gray = cv2.cvtColor(straightened, cv2.COLOR_BGR2GRAY)
    enhanced = clahe_gray(gray)

    os.makedirs(LOGS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(LOGS_DIR, "preprocessed.jpg"), enhanced)
    return enhanced


def detect_text_regions(ocr_image):
    gray = ocr_image if len(ocr_image.shape) == 2 else cv2.cvtColor(ocr_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    connected = cv2.dilate(edges, dilate_kernel, iterations=1)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / float(max(h, 1))

        if w <= 60 or h <= 20 or area < 1200:
            continue
        if aspect_ratio < 1.2 or aspect_ratio > 35:
            continue

        pad_x = max(10, int(w * 0.04))
        pad_y = max(24, int(h * 0.9))
        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(gray.shape[1], x + w + pad_x)
        y1 = min(gray.shape[0], y + h + pad_y)

        roi = gray[y0:y1, x0:x1]
        if roi.size == 0:
            continue

        regions.append({
            "bbox": (x0, y0, x1, y1),
            "image": roi,
        })
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)

    regions.sort(key=lambda item: (item["bbox"][1], item["bbox"][0]))

    if not regions:
        regions = [{
            "bbox": (0, 0, gray.shape[1], gray.shape[0]),
            "image": gray,
        }]
        cv2.rectangle(overlay, (0, 0), (gray.shape[1] - 1, gray.shape[0] - 1), (0, 255, 255), 2)

    os.makedirs(LOGS_DIR, exist_ok=True)
    cv2.imwrite(os.path.join(LOGS_DIR, "text_regions.jpg"), overlay)
    return regions


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _save_debug_image(filename: str, image: np.ndarray):
    if image is None or getattr(image, "size", 0) == 0:
        return
    _ensure_dir(DEBUG_DIR)
    cv2.imwrite(os.path.join(DEBUG_DIR, filename), image)


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return image
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def preprocessing_variants(image: np.ndarray) -> Dict[str, np.ndarray]:
    gray = _to_gray(image)
    variants: Dict[str, np.ndarray] = {}

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray)
    variants["clahe"] = clahe_img

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    adaptive = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    variants["adaptive_thresh"] = adaptive

    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(gray, -1, sharpen_kernel)
    variants["sharpen"] = sharpened

    return variants


def laplacian_variance(image: np.ndarray) -> float:
    gray = _to_gray(image)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def select_best_variant(variants: Dict[str, np.ndarray]) -> Tuple[str, np.ndarray, Dict[str, float]]:
    scores: Dict[str, float] = {}
    best_name = ""
    best_score = -1.0
    best_image = None

    for name, variant in variants.items():
        score = laplacian_variance(variant)
        scores[name] = score
        if score > best_score:
            best_score = score
            best_name = name
            best_image = variant

    if best_image is None:
        raise ValueError("No preprocessing variants available for selection")

    return best_name, best_image, scores


def _load_field_detector():
    global _FIELD_DETECTOR
    if _FIELD_DETECTOR is not None:
        return _FIELD_DETECTOR
    
    print(f"[pipeline] Checking for YOLO field model at: {YOLO_FIELD_MODEL_PATH}")
    if YOLO is None:
        print("[pipeline] ERROR: ultralytics not available; YOLO field detection disabled.")
        return None
    if not os.path.exists(YOLO_FIELD_MODEL_PATH):
        print("[pipeline] YOLO model not found, running OCR on full image")
        return None

    try:
        _FIELD_DETECTOR = YOLO(YOLO_FIELD_MODEL_PATH)
        print(f"[pipeline] YOLO model loaded successfully from {YOLO_FIELD_MODEL_PATH}")
        if hasattr(_FIELD_DETECTOR, "names"):
            print(f"[pipeline] YOLO model classes: {_FIELD_DETECTOR.names}")
    except Exception as exc:
        print(f"[pipeline] ERROR: YOLO model failed to load: {exc}")
        _FIELD_DETECTOR = None
    return _FIELD_DETECTOR


def detect_label_fields_yolo(image: np.ndarray) -> List[Dict[str, Any]]:
    model = _load_field_detector()
    if model is None:
        return []

    results = model(image, verbose=False, conf=YOLO_FIELD_CONFIDENCE_THRESHOLD, iou=YOLO_FIELD_IOU_THRESHOLD)
    detections: List[Dict[str, Any]] = []

    names = model.names if hasattr(model, "names") else {}
    for result in results:
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else names[cls_id]
            if cls_name not in FIELD_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "label": cls_name,
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
            })

    detections.sort(key=lambda det: (det["bbox"][1], det["bbox"][0]))
    
    print(f"[pipeline] YOLO detections: {len(detections)}")
    if detections:
        detected_classes = [d['label'] for d in detections]
        print(f"[pipeline] Detected classes: {', '.join(detected_classes)}")
        for det in detections:
            print(f"[pipeline] Detection: label={det['label']}, bbox={det['bbox']}, conf={det['confidence']:.2f}")
    else:
        print("[pipeline] YOLO produced zero detections.")
        
    return detections


def draw_field_boxes(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    overlay = _to_bgr(image).copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(
            overlay,
            f"{label}:{conf:.2f}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def crop_field_rois(image: np.ndarray, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rois: List[Dict[str, Any]] = []
    if image is None or getattr(image, "size", 0) == 0:
        print("[pipeline] WARNING: crop_field_rois received empty image")
        return rois

    height, width = image.shape[:2]
    for idx, det in enumerate(detections):
        x1, y1, x2, y2 = det["bbox"]
        pad = max(2, int(0.02 * max(x2 - x1, y2 - y1)))

        rx1 = max(0, x1 - pad)
        ry1 = max(0, y1 - pad)
        rx2 = min(width, x2 + pad)
        ry2 = min(height, y2 + pad)

        roi = image[ry1:ry2, rx1:rx2]
        if roi.size == 0:
            print(f"[pipeline] WARNING: ROI image for class={det['label']} is empty or zero size.")
            continue
            
        print(f"[pipeline] ROI created: class={det['label']} size={roi.shape[1]}x{roi.shape[0]}")
        rois.append({
            "field": det["label"],
            "bbox": (rx1, ry1, rx2, ry2),
            "confidence": det["confidence"],
            "image": roi,
            "skip_ocr": det["label"] == "barcode",
            "index": idx,
        })
    return rois


def save_preprocessing_debug(variants: Dict[str, np.ndarray], best_name: str, best_image: np.ndarray):
    var_dir = os.path.join(DEBUG_DIR, "variants")
    _ensure_dir(var_dir)
    for name, variant in variants.items():
        if variant is not None and getattr(variant, "size", 0) > 0:
            cv2.imwrite(os.path.join(var_dir, f"variant_{name}.jpg"), variant)
            
    best_dir = os.path.join(DEBUG_DIR, "best_image")
    _ensure_dir(best_dir)
    if best_image is not None and getattr(best_image, "size", 0) > 0:
        cv2.imwrite(os.path.join(best_dir, f"selected_best_{best_name}.jpg"), best_image)


def save_detection_debug(image: np.ndarray, detections: List[Dict[str, Any]]):
    overlay = draw_field_boxes(image, detections)
    det_dir = os.path.join(DEBUG_DIR, "detections")
    _ensure_dir(det_dir)
    if overlay is not None and getattr(overlay, "size", 0) > 0:
        cv2.imwrite(os.path.join(det_dir, "field_detections.jpg"), overlay)


def save_roi_debug(rois: List[Dict[str, Any]]):
    roi_dir = os.path.join(DEBUG_DIR, "roi")
    _ensure_dir(roi_dir)
    for roi in rois:
        field = roi.get("field", "field")
        idx = roi.get("index", 0)
        img = roi.get("image")
        if img is not None and getattr(img, "size", 0) > 0:
            cv2.imwrite(os.path.join(roi_dir, f"roi_{field}_{idx}.png"), img)





async def run_ocr_on_rois(rois: List[Dict[str, Any]], engine: str, concurrency: int = 4) -> Tuple[Dict[str, List[str]], List[Dict[str, Any]], float]:
    field_texts: Dict[str, List[str]] = {}
    blocks: List[Dict[str, Any]] = []
    confidences: List[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def process_roi(roi: Dict[str, Any]):
        field = str(roi.get("field", "")).strip()
        field_key = "capacitance" if field == "capacitor" else field
        field_texts.setdefault(field_key, [])
        if roi.get("skip_ocr"):
            return

        image = roi.get("image")
        if image is None or getattr(image, "size", 0) == 0:
            return

        print(f"[pipeline] Running OCR using engine: {engine}")
        print(f"[pipeline] ROI size: {image.shape[1]}x{image.shape[0]}")
        print(f"[pipeline] Field: {field_key}")

        async with semaphore:
            text = await asyncio.to_thread(run_ocr, image, field, engine)

        text = str(text or "").strip()
        
        if not text:
            print(f"[pipeline] OCR returned empty result.")
        else:
            print(f"[pipeline] OCR result: {repr(text)}")
            
        if text:
            field_texts[field_key].append(text)

    tasks = [asyncio.create_task(process_roi(roi)) for roi in rois]
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return field_texts, blocks, average_confidence

