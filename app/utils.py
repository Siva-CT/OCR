"""
Image validation and error handling utilities.
"""
import numpy as np
import cv2
from functools import wraps
from typing import Callable, Any

class ImageValidationError(Exception):
    """Raised when image validation fails."""
    pass

def validate_image(func: Callable) -> Callable:
    """
    Decorator to validate image inputs.
    Ensures image is not None, has correct shape, and valid dtype.
    """
    @wraps(func)
    def wrapper(image: np.ndarray, *args, **kwargs) -> Any:
        if image is None:
            raise ImageValidationError(f"{func.__name__}: Image cannot be None")
        
        if not isinstance(image, np.ndarray):
            raise ImageValidationError(f"{func.__name__}: Image must be numpy array, got {type(image)}")
        
        if len(image.shape) < 2:
            raise ImageValidationError(f"{func.__name__}: Image must be at least 2D, got shape {image.shape}")
        
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ImageValidationError(f"{func.__name__}: Image dtype must be uint8, float32 or float64, got {image.dtype}")
        
        if image.size == 0:
            raise ImageValidationError(f"{func.__name__}: Image is empty (size=0)")
        
        return func(image, *args, **kwargs)
    
    return wrapper

def safe_extract_text(func: Callable) -> Callable:
    """
    Decorator for safe text extraction.
    Catches errors and returns empty string instead of crashing.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> str:
        try:
            result = func(*args, **kwargs)
            return str(result) if result else ""
        except Exception as e:
            print(f"[TEXT_EXTRACT] Error in {func.__name__}: {e}")
            return ""
    
    return wrapper

def handle_ocr_errors(func: Callable) -> Callable:
    """
    Decorator for OCR functions.
    Standardizes error handling and logging.
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> dict:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            print(f"[OCR_ERROR] {func.__name__} failed: {e}")
            return {
                "raw_text": "",
                "blocks": [],
                "confidence": 0.0,
                "error": str(e),
            }
    
    return wrapper
