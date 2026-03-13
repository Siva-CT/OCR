import cv2


def preprocess_image(image):
    """
    Preprocess image for OCR with grayscale + CLAHE contrast enhancement.
    """
    if image is None:
        raise ValueError("Input image cannot be None")
    if len(image.shape) != 3:
        raise ValueError(f"Expected BGR image with 3 channels, got shape {image.shape}")
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    result = clahe.apply(gray)
    if result is None:
        raise ValueError("CLAHE enhancement failed")
    return result
