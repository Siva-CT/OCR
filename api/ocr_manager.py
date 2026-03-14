from __future__ import annotations

from __future__ import annotations
from ocr_engines.paddleocr_engine import run as run_paddleocr

SUPPORTED_ENGINES = {"paddleocr"}


def run_ocr(image, field_type: str, engine: str = "paddleocr") -> str:
    print("[OCR_MANAGER] Dispatching: paddleocr")
    return run_paddleocr(image, field_type)
