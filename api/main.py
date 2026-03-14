import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from pipeline import detect_and_crop_label, detect_text_regions, preprocess_for_ocr, preprocessing_variants, select_best_variant, detect_label_fields_yolo, crop_field_rois, save_preprocessing_debug, save_detection_debug, save_roi_debug, run_ocr_on_rois
from schemas import (
    detect_vendor_from_text,
    list_saved_schemas,
    map_extracted_text_to_json,
    parse_ocr_text_fields,
    parse_component_fields_from_texts,
    sanitize_structured_fields,
    save_generated_schema,
)
from generator import generate_datamatrix_string, generate_hu_number, generate_ibd_number, generate_zpl
from db import ensure_dirs, save_scan

app = FastAPI(title="Warehouse Label Scanner API Gateway")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ENGINE_URLS = {
    "paddleocr": os.environ.get("PADDLE_OCR_URL", "http://paddle-ocr-service:8001/ocr"),
}

DISPLAY_FIELDS = [
    "vendor",
    "part_number",
    "quantity",
    "vendor_lot",
    "description",
    "date_code",
    "raw_text",
    "supplier_invoice",
    "msd_level",
    "msd_date",
    "barcode",
]

# Fail fast on downstream OCR calls so the frontend isn't left hanging
HTTP_TIMEOUT = httpx.Timeout(connect=2.0, read=20.0, write=20.0, pool=20.0)
OCR_FALLBACK_TIMEOUT_SECONDS = 8.0
LOGS_DIR = "/app/logs"
REPO_KEYS_DIR = Path(__file__).resolve().parent / "keys"

jobs: Dict[str, Any] = {}

def log(msg: str):
    print(f"[API] {msg}", flush=True)


def warn_if_repo_keys_exist():
    if REPO_KEYS_DIR.exists():
        log(
            "Repository safety warning: local keys/ directory detected at "
            f"{REPO_KEYS_DIR}. Provide Google Vision credentials through "
            "GOOGLE_APPLICATION_CREDENTIALS or Secret Manager, and do not "
            "commit service account files."
        )


@app.on_event("startup")
def startup_safety_checks():
    pass


def is_missing(value: Any, field: str = "") -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip()
        if normalized in {"", "-"}:
            return True
        if field == "vendor" and normalized.upper() == "UNKNOWN":
            return True
    return False


def normalize_structured_data(data: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    normalized = dict(data or {})
    for field in DISPLAY_FIELDS:
        if is_missing(normalized.get(field), field):
            normalized[field] = "UNKNOWN" if field == "vendor" else "-"
    return normalized

def normalize_ocr_output(payload: Dict[str, Any]) -> str:
    raw_text = payload.get("raw_text", "")
    if isinstance(raw_text, str) and raw_text.strip():
        return raw_text.strip()

    blocks = payload.get("blocks", [])
    block_lines = []
    if isinstance(blocks, list):
        for block in blocks:
            if isinstance(block, dict):
                text = str(block.get("text", "")).strip()
                if text:
                    block_lines.append(text)
    if block_lines:
        return "\n".join(block_lines)

    text_payload = payload.get("text", [])
    lines = []
    pages = text_payload if isinstance(text_payload, list) else [text_payload]
    for page in pages:
        page_items = page if isinstance(page, list) else [page]
        for item in page_items:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            text_info = item[1]
            if isinstance(text_info, (list, tuple)) and text_info:
                value = str(text_info[0]).strip()
            else:
                value = str(text_info).strip()
            if value:
                lines.append(value)
    return "\n".join(lines)


def empty_ocr_result() -> Dict[str, Any]:
    return {
        "raw_text": "",
        "blocks": [],
        "confidence": 0.0,
    }


def normalize_engine_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = dict(payload or {})
    blocks = data.get("blocks", [])
    if not isinstance(blocks, list):
        blocks = []

    confidence_values = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        try:
            confidence_values.append(float(block.get("confidence", 0.0)))
        except Exception:
            continue

    return {
        "raw_text": normalize_ocr_output(data),
        "blocks": blocks,
        "confidence": (sum(confidence_values) / len(confidence_values)) if confidence_values else 0.0,
    }


def extraction_score(raw_text: str) -> tuple[int, int]:
    text = str(raw_text or "").strip()
    if not text:
        return 0, 0

    vendor_hint = detect_vendor_from_text(text)
    parsed = parse_ocr_text_fields(text, vendor_hint=vendor_hint)
    structured = map_extracted_text_to_json(text, vendor_hint=vendor_hint)

    score = 0
    if vendor_hint and vendor_hint != "UNKNOWN":
        score += 2
    if parsed.get("part", "-") != "-":
        score += 4
    if parsed.get("qty", "-") != "-":
        score += 3
    if parsed.get("ven_lot_no", "-") != "-":
        score += 3
    if structured.get("date_code", "-") != "-":
        score += 2

    return score, len(text)


def ocr_result_score(payload: Dict[str, Any]) -> tuple[int, int, int]:
    field_score, text_length = extraction_score(str(payload.get("raw_text", "") or ""))
    confidence = int(float(payload.get("confidence", 0.0) or 0.0) * 1000)
    return field_score, text_length, confidence


async def call_ocr_engine(engine: str, file_bytes: bytes) -> Dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            res = await client.post(
                ENGINE_URLS[engine],
                files={"file": ("processed.jpg", file_bytes, "image/jpeg")},
            )
        if res.status_code == 200:
            return normalize_engine_payload(res.json())
        log(f"OCR engine {engine} returned {res.status_code}: {res.text[:300]}")
    except Exception as e:
        log(f"Engine {engine} failed: {e}")
    return empty_ocr_result()


async def run_selected_engine(engine: str, file_bytes: bytes) -> Dict[str, Any]:
    return await call_ocr_engine(engine, file_bytes)


def prepare_region_for_ocr(image: np.ndarray, engine: str) -> np.ndarray:
    region = image.copy()
    height, width = region.shape[:2]

    max_width = 2200 if engine == "paddleocr" else 2600
    if width > max_width:
        scale = max_width / float(width)
        region = cv2.resize(region, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
        height, width = region.shape[:2]

    min_height = 72
    if height < min_height:
        scale = min_height / float(max(height, 1))
        region = cv2.resize(region, (max(1, int(width * scale)), min_height), interpolation=cv2.INTER_CUBIC)

    return cv2.copyMakeBorder(region, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=255)


async def run_ocr_on_image(engine: str, image: np.ndarray) -> Dict[str, Any]:
    if image is None or getattr(image, "size", 0) == 0:
        return empty_ocr_result()

    prepared_image = prepare_region_for_ocr(image, engine)
    ok, encoded = cv2.imencode(".jpg", prepared_image)
    if not ok:
        log("Skipping OCR image: encoding failed")
        return empty_ocr_result()

    return await run_selected_engine(engine, encoded.tobytes())


async def run_ocr_on_regions(engine: str, regions: list[Dict[str, Any]], timeout_seconds: float = OCR_FALLBACK_TIMEOUT_SECONDS) -> Dict[str, Any]:
    if not regions:
        return empty_ocr_result()

    concurrency = 2 if engine == "paddleocr" else 4
    semaphore = asyncio.Semaphore(concurrency)
    region_limit = 4 if engine == "paddleocr" else 8

    async def process_region(index: int, region: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        image = region.get("image")
        if image is None or getattr(image, "size", 0) == 0:
            return index, empty_ocr_result()

        image = prepare_region_for_ocr(image, engine)

        ok, encoded = cv2.imencode(".jpg", image)
        if not ok:
            log(f"Skipping OCR region {index}: encoding failed")
            return index, empty_ocr_result()

        async with semaphore:
            payload = await run_selected_engine(engine, encoded.tobytes())

        cleaned_lines = [line.strip() for line in str(payload.get("raw_text", "") or "").splitlines() if line.strip()]
        blocks = []
        for block in payload.get("blocks", []):
            if not isinstance(block, dict):
                continue
            enriched_block = dict(block)
            enriched_block.setdefault("region_index", index)
            blocks.append(enriched_block)

        return index, {
            "raw_text": "\n".join(cleaned_lines),
            "blocks": blocks,
            "confidence": float(payload.get("confidence", 0.0) or 0.0),
        }

    tasks = [
        asyncio.create_task(process_region(index, region))
        for index, region in enumerate(regions[:region_limit])
    ]
    done, pending = await asyncio.wait(tasks, timeout=timeout_seconds)

    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
        log(f"OCR fallback timed out with {len(pending)} region(s) still pending")

    ordered_lines = []
    confidences = []
    merged_blocks = []

    for task in done:
        if task.cancelled():
            continue

        try:
            result = task.result()
        except Exception as exc:
            log(f"OCR region task failed: {exc}")
            continue

        _, payload = result
        text = str(payload.get("raw_text", "") or "")
        confidence = float(payload.get("confidence", 0.0) or 0.0)
        if text:
            ordered_lines.extend([line for line in text.splitlines() if line.strip()])
            if confidence:
                confidences.append(confidence)
        merged_blocks.extend(payload.get("blocks", []))

    deduped_lines = []
    seen_lines = set()
    for line in ordered_lines:
        normalized_line = line.strip()
        if normalized_line and normalized_line not in seen_lines:
            deduped_lines.append(normalized_line)
            seen_lines.add(normalized_line)

    merged_text = "\n".join(deduped_lines)
    average_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return {
        "raw_text": merged_text,
        "blocks": merged_blocks,
        "confidence": average_confidence,
    }


def build_ocr_candidates(engine: str, original_img: np.ndarray, cropped_img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    candidates: list[tuple[str, np.ndarray]] = []
    processed_cropped = preprocess_for_ocr(cropped_img)
    processed_original = preprocess_for_ocr(original_img)

    if engine == "tesseract":
        ordered_candidates = [
            ("original_color", original_img),
            ("cropped_color", cropped_img),
            ("original_preprocessed", processed_original),
            ("cropped_preprocessed", processed_cropped),
        ]
    else:
        ordered_candidates = [
            ("cropped_color", cropped_img),
            ("original_color", original_img),
            ("cropped_preprocessed", processed_cropped),
            ("original_preprocessed", processed_original),
        ]

    for name, image in ordered_candidates:
        if image is None or getattr(image, "size", 0) == 0:
            continue
        candidates.append((name, image))

    return candidates


async def run_best_ocr_pass(engine: str, original_img: np.ndarray, cropped_img: np.ndarray) -> Dict[str, Any]:
    best_result = empty_ocr_result()
    best_score = ocr_result_score(best_result)
    best_candidate_name = "none"

    for candidate_name, candidate_image in build_ocr_candidates(engine, original_img, cropped_img):
        candidate_result = await run_ocr_on_image(engine, candidate_image)
        candidate_score = ocr_result_score(candidate_result)
        log(
            f"Stage: OCR candidate {candidate_name} primary "
            f"score={candidate_score[0]} text_len={candidate_score[1]}"
        )

        if candidate_score[1] < 10 or candidate_score[0] < 4:
            text_regions = detect_text_regions(candidate_image)
            log(f"Stage: text region detection count ({candidate_name}) = {len(text_regions)}")
            fallback_result = await run_ocr_on_regions(engine, text_regions)
            fallback_score = ocr_result_score(fallback_result)
            if fallback_score > candidate_score:
                candidate_result = fallback_result
                candidate_score = fallback_score
                log(f"Stage: OCR candidate {candidate_name} improved via region fallback")

        if candidate_score > best_score:
            best_result = candidate_result
            best_score = candidate_score
            best_candidate_name = candidate_name

        if candidate_score[0] >= 10:
            break

    log(
        f"Stage: best OCR candidate = {best_candidate_name} "
        f"(score={best_score[0]}, text_len={best_score[1]})"
    )
    return best_result


def fallback_payload(engine_used: str = "fallback") -> dict:
    parsed = {
        "ibd_no": "-",
        "barcode": "-",
        "part": "-",
        "qty": "-",
        "vendor": "UNKNOWN",
        "supplier_invoice": "-",
        "ven_lot_no": "-",
        "msd_level": "-",
        "msd_date": "-",
    }
    structured_data = {
        "vendor": parsed["vendor"],
        "part_number": "-",
        "quantity": "-",
        "capacitance": "-",
        "voltage": "-",
        "date_code": "-",
        "lot": "-",
        "vendor_lot": "-",
        "description": "-",
        "raw_text": "-",
        "barcode": "-",
        "hu": parsed["barcode"],
        "ibd": parsed["ibd_no"],
        "supplier_invoice": parsed["supplier_invoice"],
        "msd_level": parsed["msd_level"],
        "msd_date": parsed["msd_date"],
        "datamatrix": "",
        "zpl": "",
        "engine_used": engine_used,
    }
    return {
        "raw_text": "",
        "blocks": [],
        "parsed": parsed,
        "status": "success",
        "structured_data": structured_data,
        "meta": {
            "engine_used": engine_used,
            "confidence": 0.0,
            "processing_time": "0.00s",
            "schema_needed": False,
            "pending_schema_id": "",
        }
    }


def build_response_payload(
    raw_text: str,
    blocks: list[Dict[str, Any]],
    parsed: Dict[str, Any],
    structured_data: Dict[str, Any],
    engine_used: str,
    confidence: float,
    processing_time: float,
    schema_needed: bool = False,
    pending_schema_id: str = ""
) -> Dict[str, Any]:
    return {
        "raw_text": raw_text,
        "blocks": blocks,
        "parsed": parsed,
        "status": "success",
        "structured_data": structured_data,
        "meta": {
            "engine_used": engine_used,
            "confidence": confidence,
            "processing_time": f"{processing_time:.2f}s",
            "schema_needed": schema_needed,
            "pending_schema_id": pending_schema_id,
        },
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ocr-health")
async def ocr_health():
    results = {}
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        for name, url in ENGINE_URLS.items():
            try:
                r = await client.get(url.replace("/ocr", "/health"))
                results[name] = r.status_code
            except Exception as e:
                results[name] = str(e)
    return results


@app.get("/schemas")
def get_schemas():
    return {"schemas": list_saved_schemas()}


@app.post("/scan")
async def scan_label(
    file: UploadFile = File(...)
):
    try:
        start_time = time.time()
        selected_engine = "paddleocr"
        log(f"Running OCR using engine: {selected_engine}")

        contents = await file.read()
        image_hash = hashlib.sha256(contents).hexdigest() if contents else ""
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return fallback_payload("invalid_image")

        log("Stage: capture")
        log(f"Received image resolution: {img.shape[1]}x{img.shape[0]}")

        os.makedirs(LOGS_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(LOGS_DIR, "last_capture.jpg"), img)

        variants = preprocessing_variants(img)
        best_name, best_image, sharpness_scores = select_best_variant(variants)
        save_preprocessing_debug(variants, best_name, best_image)
        log("Stage: preprocessing variants " + ", ".join([f"{k}={v:.2f}" for k, v in sharpness_scores.items()]))
        log(f"Stage: selected variant = {best_name}")

        if len(best_image.shape) == 2:
            best_for_detection = cv2.cvtColor(best_image, cv2.COLOR_GRAY2BGR)
        else:
            best_for_detection = best_image

        detections = detect_label_fields_yolo(best_for_detection)
        save_detection_debug(best_for_detection, detections)
        log(f"Stage: YOLO field detections = {len(detections)}")

        if len(detections) == 0:
            log("YOLO produced zero detections. Running OCR on full image for debugging.")
            fallback_result = await run_ocr_on_image(selected_engine, best_image)
            rois = []
            field_texts = {}
            ocr_blocks = fallback_result.get("blocks", [])
            confidence = fallback_result.get("confidence", 0.0)
            combined_text = fallback_result.get("raw_text", "")
            ordered_lines = [line.strip() for line in combined_text.splitlines() if line.strip()]
        else:
            rois = crop_field_rois(best_image, detections)
            save_roi_debug(rois)
            log(f"Stage: ROI crops = {len(rois)}")

            field_texts, ocr_blocks, confidence = await run_ocr_on_rois(rois, selected_engine)

            ordered_lines = []
            seen_lines = set()
            for values in field_texts.values():
                for raw_line in values:
                    for line in str(raw_line).splitlines():
                        cleaned = line.strip()
                        if cleaned and cleaned not in seen_lines:
                            ordered_lines.append(cleaned)
                            seen_lines.add(cleaned)
            combined_text = "\n".join(ordered_lines)
        log(f"Stage: OCR text lines = {len(ordered_lines)}")
        
        structured_data = map_extracted_text_to_json(combined_text)
        
        for key, val in structured_data.items():
            log(f"Parsed {key}: {val}")

        print("[FINAL PARSED DATA]", structured_data)

        parsed_fields = {
            "ibd_no": "-",
            "barcode": "-",
            "part": structured_data.get("part_number", "-"),
            "qty": structured_data.get("quantity", "-"),
            "vendor": structured_data.get("vendor", "UNKNOWN"),
            "supplier_invoice": "-",
            "ven_lot_no": structured_data.get("vendor_lot", "-"),
            "msd_level": "-",
            "msd_date": "-",
        }

        hu_number = generate_hu_number()
        ibd_number = generate_ibd_number()
        msd_level = "0"
        vendor_code = structured_data.get("vendor", "UNK")
        datamatrix_str = generate_datamatrix_string(
            hu_number,
            structured_data.get("part_number", "-"),
            structured_data.get("vendor_lot", "-"),
            structured_data.get("quantity", "-"),
            msd_level,
            vendor_code,
        )
        zpl_code = generate_zpl(
            datamatrix_str,
            ibd_number,
            structured_data.get("part_number", "-"),
            structured_data.get("description", "-"),
            structured_data.get("quantity", "-"),
            "EA",
            hu_number,
            structured_data.get("vendor", "UNKNOWN"),
            "",
            structured_data.get("vendor_lot", "-"),
            msd_level,
            structured_data.get("date_code", "-"),
        )

        structured_data["hu"] = hu_number
        structured_data["ibd"] = ibd_number
        structured_data["engine_used"] = selected_engine
        structured_data["datamatrix"] = datamatrix_str
        structured_data["zpl"] = zpl_code

        processing_time = time.time() - start_time

        try:
            saved_schema = save_generated_schema(
                structured_data.get("vendor", "UNKNOWN"),
                structured_data.get("raw_text", ""),
                parsed_fields,
            )
            if saved_schema:
                log(
                    f"Stage: schema saved for {saved_schema.get('vendor', 'UNKNOWN')} "
                    f"with score {saved_schema.get('match_score', 0)}"
                )
        except Exception as e:
            print(f"Failed to persist generated schema: {e}")

        ensure_dirs()
        record = {
            **structured_data,
            "vendor_lot": structured_data.get("vendor_lot", "-"),
            "engine_used": structured_data["engine_used"],
            "processing_time": f"{processing_time:.2f}s",
            "raw_text": structured_data.get("raw_text", ""),
            "image_hash": image_hash,
        }
        try:
            save_scan(record)
        except Exception as e:
            print(f"Failed to save scan: {e}")

        debug_dirs = [
            "/app/logs/debug/variants",
            "/app/logs/debug/best_image",
            "/app/logs/debug/detections",
            "/app/logs/debug/roi"
        ]
        for d in debug_dirs:
            if not os.path.exists(d) or not os.listdir(d):
                log(f"WARNING: Debug directory {d} is empty or does not exist.")

        response_payload = build_response_payload(
            raw_text=structured_data.get("raw_text", ""),
            blocks=ocr_blocks,
            parsed=parsed_fields,
            structured_data=structured_data,
            engine_used=selected_engine,
            confidence=confidence,
            processing_time=processing_time,
        )
        
        response_payload["debug"] = {
            "detections_count": len(detections),
            "roi_count": len(rois),
            "ocr_engine": selected_engine,
            "ocr_text": structured_data.get("raw_text", "")
        }
        
        log(f"Response ready in {processing_time:.2f}s using {selected_engine}")
        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unhandled scan error: {e}")
        return fallback_payload()




