import cv2
import pytesseract
from .base_ocr import BaseOCR
from .preprocess import prepare_image_for_ocr

# PSM 11 = sparse text (best for labels with scattered/varied field layouts)
# preserve_interword_spaces keeps field spacing intact
_TESS_CONFIG = (
    "--oem 3 "
    "--psm 11 "
    "-c preserve_interword_spaces=1 "
    "-c tessedit_char_whitelist="
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "/:-., "
)


class TesseractOCREngine(BaseOCR):
    def extract_text(self, image) -> str:
        """Extracts text using Tesseract with label-optimized configuration."""
        if image is None:
            raise ValueError("Input image cannot be None")
        
        prepared_img = prepare_image_for_ocr(image)
        if prepared_img is None:
            raise ValueError("Image preparation failed")
        
        # pytesseract works best with BGR; convert from grayscale if needed
        if len(prepared_img.shape) == 2:
            image_for_ocr = cv2.cvtColor(prepared_img, cv2.COLOR_GRAY2BGR)
        else:
            image_for_ocr = prepared_img
        text = pytesseract.image_to_string(image_for_ocr, config=_TESS_CONFIG)
        # Post-process common OCR mistakes on electronic labels
        text = text.replace("AVNETASIAPTE", "AVNET ASIA PTE")
        text = text.replace("AVNST", "AVNET")
        text = text.replace("PTR", "PTE")
        # Normalize whitespace
        import re
        text = re.sub(r"\s+", " ", text).strip()
        return text or ""