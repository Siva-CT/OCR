import os

import cv2
import numpy as np

MAX_WIDTH = 1280
LOGS_DIR = "/app/logs"


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
