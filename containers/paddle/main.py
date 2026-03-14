import os
import asyncio
import threading

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_use_mkldnn", "0")

from paddleocr import PaddleOCR

app = FastAPI(title="PaddleOCR Service")
OCR_REQUEST_TIMEOUT_SECONDS = 10.0
OCR_REQUEST_LOCK = threading.Lock()

# Load the OCR model once when the container starts.
ocr = PaddleOCR(
    use_angle_cls=False,
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="en_PP-OCRv5_mobile_rec",
    text_det_limit_side_len=960,
    device="cpu",
    enable_hpi=False,
    enable_mkldnn=False,
    cpu_threads=2,
)


def _empty_response():
    return {
        "raw_text": "",
        "blocks": [],
    }


def _normalize_result(result):
    extracted_text = []
    raw_lines = []

    if not result:
        return extracted_text, ""

    pages = result if isinstance(result, list) else [result]
    for page in pages:
        if page is None:
            continue

        if isinstance(page, dict):
            texts = page.get("rec_texts", []) or []
            scores = page.get("rec_scores", []) or []
            boxes = page.get("rec_polys", []) or page.get("dt_polys", []) or []

            for index, value in enumerate(texts):
                text = str(value).strip()
                if not text:
                    continue

                confidence = 0.0
                if index < len(scores):
                    try:
                        confidence = float(scores[index])
                    except Exception:
                        confidence = 0.0

                box = []
                if index < len(boxes):
                    candidate_box = boxes[index]
                    box = candidate_box.tolist() if hasattr(candidate_box, "tolist") else candidate_box

                # Deduplicate: don't add same text twice
                if text and not any(text == b["text"] for b in extracted_text):
                    extracted_text.append({
                        "text": text,
                        "confidence": confidence,
                        "box": box,
                    })
                    raw_lines.append(text)
            continue

        lines = page if isinstance(page, list) else [page]
        for line in lines:
            if not isinstance(line, (list, tuple)) or len(line) < 2:
                continue

            box = line[0]
            text_info = line[1]
            if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                text = str(text_info[0]).strip()
                try:
                    confidence = float(text_info[1])
                except Exception:
                    confidence = 0.0
            else:
                text = str(text_info).strip()
                confidence = 0.0

            if text:
                extracted_text.append({
                    "text": text,
                    "confidence": confidence,
                    "box": box
                })
                raw_lines.append(text)

    return extracted_text, "\n".join(raw_lines)


def _run_ocr(image: np.ndarray):
    with OCR_REQUEST_LOCK:
        result = ocr.ocr(image)
    return _normalize_result(result)


@app.post("/ocr")
async def process_image(file: UploadFile = File(...)):
    if not str(file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image")

    try:
        extracted_text, raw_text = await asyncio.wait_for(
            asyncio.to_thread(_run_ocr, img),
            timeout=OCR_REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        print("[PADDLE] OCR inference timed out", flush=True)
        return _empty_response()
    except Exception as exc:
        print(f"[PADDLE] OCR inference failed: {exc}", flush=True)
        return _empty_response()

    return {
        "raw_text": raw_text,
        "blocks": extracted_text,
    }

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "paddleocr"}
