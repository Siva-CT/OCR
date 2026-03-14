from __future__ import annotations

import cv2
import numpy as np

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

_OCR = None


def _get_ocr() -> PaddleOCR:
    global _OCR
    if _OCR is not None:
        return _OCR
    if PaddleOCR is None:
        raise RuntimeError("PaddleOCR is not available")
    _OCR = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
    return _OCR


def _to_rgb(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _extract_text(result) -> str:
    lines = []
    for page in result or []:
        for item in page or []:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text_info = item[1]
            if isinstance(text_info, (list, tuple)) and text_info:
                text = str(text_info[0])
            else:
                text = str(text_info)
            text = text.strip()
            if text:
                lines.append(text)
    return "\n".join(lines).strip()


def run(image: np.ndarray, field_type: str) -> str:
    print("[ENGINE] Running PaddleOCR")
    if getattr(image, "size", 0) == 0:
        return ""
    try:
        ocr = _get_ocr()
        rgb = _to_rgb(image)
        result = ocr.ocr(rgb, cls=True)
    except Exception:
        return ""
    return _extract_text(result)
