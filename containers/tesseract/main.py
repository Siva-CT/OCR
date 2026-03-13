import asyncio
import threading

import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

app = FastAPI(title="Tesseract OCR Service")
OCR_REQUEST_TIMEOUT_SECONDS = 10.0
OCR_REQUEST_LOCK = threading.Lock()

# Lightweight config: English only, OEM 1 (LSTM), PSM 4 (single column text)
# A narrow whitelist helps label-style OCR stay fast and reduces noise.
TESS_CONFIG = "--oem 1 --psm 4 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"


def _empty_response():
    return {
        "raw_text": "",
        "blocks": [],
    }


def _preprocess_image(img_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)

    target_width = int(os.environ.get("TESSERACT_TARGET_WIDTH", "1200"))
    height, width = blurred.shape
    if width and width != target_width:
        scale = target_width / float(width)
        target_height = max(1, int(height * scale))
        return cv2.resize(blurred, (target_width, target_height), interpolation=cv2.INTER_CUBIC)
    return blurred


def _run_tesseract(img_bgr: np.ndarray):
    with OCR_REQUEST_LOCK:
        processed_img = _preprocess_image(img_bgr)
        pil_img = Image.fromarray(processed_img)

        raw_text = pytesseract.image_to_string(pil_img, config=TESS_CONFIG).strip()
        data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT, config=TESS_CONFIG)

        extracted_text = []
        raw_text_parts = []
        for index in range(len(data["text"])):
            text = str(data["text"][index]).strip()
            if not text:
                continue

            try:
                confidence = max(0.0, float(data["conf"][index]) / 100.0)
            except Exception:
                confidence = 0.0

            x = int(data["left"][index])
            y = int(data["top"][index])
            w = int(data["width"][index])
            h = int(data["height"][index])
            box = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

            extracted_text.append({
                "text": text,
                "confidence": confidence,
                "box": box,
            })
            raw_text_parts.append(text)

        if not raw_text:
            raw_text = "\n".join(raw_text_parts)

        return {
            "raw_text": raw_text,
            "blocks": extracted_text,
        }

@app.post("/ocr")
async def process_image(file: UploadFile = File(...)):
    if not str(file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    try:
        nparr = np.frombuffer(contents, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("OpenCV failed to decode image")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {e}")

    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_tesseract, img_bgr),
            timeout=OCR_REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print("[TESSERACT] OCR inference timed out", flush=True)
        return _empty_response()
    except Exception as exc:
        print(f"[TESSERACT] OCR inference failed: {exc}", flush=True)
        return _empty_response()

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "tesseract"}
