import cv2
import numpy as np
import os

# Configurable constants
SEP_THRESHOLD_FACTOR = float(os.environ.get("SEP_THRESHOLD_FACTOR", "0.02"))
MIN_SEP_THRESHOLD = 1.0
MIN_SEGMENT_HEIGHT = int(os.environ.get("MIN_SEGMENT_HEIGHT", "120"))
MIN_LABEL_WIDTH_RATIO = float(os.environ.get("MIN_LABEL_WIDTH_RATIO", "0.6"))


def detect_labels(image):
    """
    Row-projection based stacked label detector.
    Splits vertically stacked reel-strip labels by finding near-zero
    pixel-density rows (blank separators between labels).
    Returns a list of cropped label images sorted top-to-bottom.
    Falls back to the full image if no segments are found.
    """
    img_h, img_w = image.shape[:2]

    # --- 1. Grayscale conversion ---
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- 2. Adaptive threshold to highlight text pixels ---
    thresh = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        10
    )

    # --- 3. Horizontal projection: sum white pixels per row ---
    projection = np.sum(thresh, axis=1).astype(np.float32)

    # --- 4. Smooth to suppress single-row noise spikes ---
    projection = cv2.GaussianBlur(
        projection.reshape(-1, 1), (1, 21), 0
    ).flatten()

    # --- 5. Mark separator rows (near-zero energy) ---
    sep_threshold = max(projection.max() * SEP_THRESHOLD_FACTOR, MIN_SEP_THRESHOLD)
    separators = projection < sep_threshold

    # --- 6. Walk rows to collect contiguous label bands ---
    segments = []
    start = None

    for row, is_gap in enumerate(separators):
        if not is_gap and start is None:
            start = row           # entering label band
        elif is_gap and start is not None:
            end = row             # leaving label band
            if end - start > MIN_SEGMENT_HEIGHT:
                segments.append((start, end))
            start = None

    # Capture final segment if it reaches the image bottom
    if start is not None and img_h - start > MIN_SEGMENT_HEIGHT:
        segments.append((start, img_h))

    # --- 7. Build crops, filter by minimum dimensions ---
    min_width = img_w * MIN_LABEL_WIDTH_RATIO
    label_crops = []

    for (y1, y2) in segments:
        crop = image[y1:y2, 0:img_w]
        if crop.shape[1] >= min_width:
            label_crops.append(crop)

    # --- 8. Safe fallback ---
    if not label_crops:
        print("[label_detector] No segments found — returning full image")
        return [image]

    print(f"Detected label regions: {len(label_crops)}")
    return label_crops