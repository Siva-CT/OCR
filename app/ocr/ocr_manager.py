from .tesseract_ocr import TesseractOCREngine
from .google_vision import GoogleVisionOCREngine

# Lazy-load engines to avoid startup failures
ENGINES = {}

def _init_tesseract():
    try:
        return TesseractOCREngine()
    except Exception as e:
        print(f"WARNING: Tesseract engine initialization failed: {e}")
        return None

def _init_google_vision():
    try:
        return GoogleVisionOCREngine()
    except Exception as e:
        print(f"WARNING: Google Vision engine initialization failed: {e}")
        return None

def get_ocr_engine(engine: str):
    """
    Returns the cached instance of the requested OCR engine.
    Supports: tesseract, google_vision
    Lazy-loads engines on first request.
    """
    key = engine.lower()
    
    # Initialize engine on first request
    if key == "tesseract" and key not in ENGINES:
        ENGINES["tesseract"] = _init_tesseract()
    elif key == "google_vision" and key not in ENGINES:
        ENGINES["google_vision"] = _init_google_vision()
    
    if key in ENGINES and ENGINES[key] is not None:
        return ENGINES[key]

    raise ValueError(f"Unsupported OCR engine: {engine}. Valid options: tesseract, google_vision")