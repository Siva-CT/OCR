import cv2
import numpy as np


def prepare_image_for_ocr(image):
    """
    Label-optimized preprocessing pipeline for Tesseract.
    Accepts BGR or grayscale images safely.
    Returns a binarized, upscaled grayscale image.
    """
    # Convert to grayscale safely
    if len(image.shape) == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Histogram equalization to increase local contrast
    gray = cv2.equalizeHist(gray)

    # Light blur to reduce noise
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Otsu threshold for clean binary image
    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Slight dilation to strengthen thin characters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Upscale 2.5x for better OCR recognition on small text
    scaled = cv2.resize(
        thresh,
        None,
        fx=2.5,
        fy=2.5,
        interpolation=cv2.INTER_CUBIC
    )

    return scaled
