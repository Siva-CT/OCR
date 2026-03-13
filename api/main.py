import asyncio
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from pipeline import detect_and_crop_label, detect_text_regions, preprocess_for_ocr
from schemas import (
    detect_vendor_from_text,
    list_saved_schemas,
    map_extracted_text_to_json,
    parse_ocr_text_fields,
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
    "google_vision": os.environ.get("GOOGLE_VISION_URL", "http://google-vision-service:8003/ocr"),
    "tesseract": os.environ.get("TESSERACT_OCR_URL", "http://tesseract-ocr-service:8002/ocr"),
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
GOOGLE_APPLICATION_CREDENTIALS = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    "/secrets/google-vision-key",
)
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
    warn_if_repo_keys_exist()


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


def _run_google_vision(file_bytes: bytes) -> Dict[str, Any]:
    if not os.path.exists(GOOGLE_APPLICATION_CREDENTIALS):
        log(
            "Vision creds missing; set GOOGLE_APPLICATION_CREDENTIALS or mount "
            f"a secret at {GOOGLE_APPLICATION_CREDENTIALS}. Skipping vision."
        )
        return empty_ocr_result()

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GOOGLE_APPLICATION_CREDENTIALS
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=file_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        log(f"GCP Vision error: {response.error.message}")
        return empty_ocr_result()

    raw_text = response.full_text_annotation.text if response.full_text_annotation else ""
    blocks = []
    for annotation in response.text_annotations[1:]:
        vertices = [[vertex.x, vertex.y] for vertex in annotation.bounding_poly.vertices]
        blocks.append({
            "text": annotation.description,
            "confidence": 1.0,
            "box": vertices,
        })

    return normalize_engine_payload({
        "raw_text": raw_text,
        "blocks": blocks,
    })


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
    return {
        "raw_text": "",
        "blocks": [],
        "parsed": parsed,
        "status": "success",
        "structured_data": {
            "vendor": parsed["vendor"],
            "part_number": "-",
            "quantity": "-",
            "vendor_lot": parsed["ven_lot_no"],
            "description": "-",
            "date_code": "-",
            "raw_text": "-",
            "hu": parsed["barcode"],
            "ibd": parsed["ibd_no"],
            "supplier_invoice": parsed["supplier_invoice"],
            "msd_level": parsed["msd_level"],
            "msd_date": parsed["msd_date"],
            "datamatrix": "",
            "zpl": "",
            "engine_used": engine_used
        },
        "meta": {
            "engine_used": engine_used,
            "confidence": 0.0,
            "processing_time": "0.00s",
            "schema_needed": False,
            "pending_schema_id": "",
        }
    }


def build_response_payload(raw_text: str, blocks: list[Dict[str, Any]], parsed: Dict[str, Any], structured_data: Dict[str, Any], engine_used: str, confidence: float, processing_time: float, schema_needed: bool = False, pending_schema_id: str = "") -> Dict[str, Any]:
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
    engine: str = Form(default="paddleocr"),
    file: UploadFile = File(...)
):
    try:
        start_time = time.time()
        selected_engine = engine if engine in ENGINE_URLS else "paddleocr"
        
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
            
        cropped_img, labels_detected = detect_and_crop_label(img)
        cv2.imwrite(os.path.join(LOGS_DIR, "cropped_label.jpg"), cropped_img)
        log(f"Stage: label detection {'OK' if labels_detected else 'fallback to full frame'}")

        engine_used = selected_engine
        confidence = 0.0

        flat_json: Dict[str, Any] = {}
        parsed_fields = {
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
        ocr_result = empty_ocr_result()

        try:
            ocr_result = await run_best_ocr_pass(selected_engine, img, cropped_img)
            confidence = float(ocr_result.get("confidence", 0.0) or 0.0)
            normalized_ocr_text = str(ocr_result.get("raw_text", "") or "").strip()
            log(f"Stage: selected OCR text length = {len(normalized_ocr_text)}")
            log(f"Stage: OCR engine used = {selected_engine}")
        except Exception as e:
            log(f"Stage: OCR engine failed = {selected_engine}: {e}")

        normalized_ocr_text = str(ocr_result.get("raw_text", "") or "").strip()
        ocr_blocks = ocr_result.get("blocks", [])
        print("OCR TEXT:", flush=True)
        print(normalized_ocr_text, flush=True)
        print("OCR TEXT LINES:", flush=True)
        for line in [line.strip() for line in normalized_ocr_text.splitlines() if line.strip()]:
            print(f"- {line}", flush=True)
        log(f"Stage: OCR text length = {len(normalized_ocr_text)}")

        if normalized_ocr_text:
            vendor_hint = detect_vendor_from_text(normalized_ocr_text)
            log(f"Stage: vendor detection = {vendor_hint or 'UNKNOWN'}")
            parsed_fields = parse_ocr_text_fields(
                normalized_ocr_text,
                vendor_hint=vendor_hint
            )
            flat_json = map_extracted_text_to_json(
                normalized_ocr_text,
                vendor_hint=vendor_hint,
            )
            log("Stage: line/regex field extraction completed")
        else:
            log("Stage: vendor detection = UNKNOWN")
            log("Stage: extraction skipped because OCR text was empty")

        if not flat_json:
            detected_vendor = detect_vendor_from_text(normalized_ocr_text) if normalized_ocr_text else "UNKNOWN"
            flat_json = {
                "vendor": detected_vendor,
                "part_number": "-",
                "quantity": "-",
                "vendor_lot": "-",
                "description": normalized_ocr_text[:160] if normalized_ocr_text else "-",
                "date_code": "-",
                "raw_text": normalized_ocr_text or "-",
                "supplier_invoice": "-",
                "msd_level": "-",
                "msd_date": "-",
                "barcode": "-",
            }

        processing_time = time.time() - start_time

        flat_json = normalize_structured_data(flat_json)
        flat_json = sanitize_structured_fields(flat_json)
        flat_json["raw_text"] = normalized_ocr_text or "-"
        flat_json["supplier_invoice"] = flat_json.get("supplier_invoice", "-") or "-"
        flat_json["msd_level"] = flat_json.get("msd_level", "-") or "-"
        flat_json["msd_date"] = flat_json.get("msd_date", "-") or "-"
        flat_json["barcode"] = flat_json.get("barcode", "-") or "-"

        log(f"Stage: vendor detection final = {flat_json.get('vendor', 'UNKNOWN')}")

        parsed_fields["vendor"] = flat_json.get("vendor", "UNKNOWN")
        if parsed_fields.get("part", "-") == "-":
            parsed_fields["part"] = flat_json.get("part_number", "-")
        if parsed_fields.get("qty", "-") == "-":
            parsed_fields["qty"] = flat_json.get("quantity", "-")
        if parsed_fields.get("ven_lot_no", "-") == "-":
            parsed_fields["ven_lot_no"] = flat_json.get("vendor_lot", "-")
        if parsed_fields.get("supplier_invoice", "-") == "-":
            parsed_fields["supplier_invoice"] = flat_json.get("supplier_invoice", "-")
        if parsed_fields.get("msd_level", "-") == "-":
            parsed_fields["msd_level"] = flat_json.get("msd_level", "-")
        if parsed_fields.get("msd_date", "-") == "-":
            parsed_fields["msd_date"] = flat_json.get("msd_date", "-")

        hu_number = parsed_fields.get("barcode", "-")
        if hu_number in {"", "-"}:
            hu_number = generate_hu_number()
        ibd_number = parsed_fields.get("ibd_no", "-")
        if ibd_number in {"", "-"}:
            ibd_number = generate_ibd_number()
        log(f"Generated HU: {hu_number}")
        log(f"Generated IBD: {ibd_number}")
        msd_level = parsed_fields.get("msd_level", "-")
        if msd_level in {"", "-"}:
            msd_level = "0"
        vendor_code = flat_json.get("vendor", "UNK")
        datamatrix_str = generate_datamatrix_string(
            hu_number,
            flat_json.get("part_number", "-"),
            flat_json.get("vendor_lot", "-"),
            flat_json.get("quantity", "-"),
            msd_level,
            vendor_code
        )
        zpl_code = generate_zpl(
            datamatrix_str,
            ibd_number,
            flat_json.get("part_number", "-"),
            flat_json.get("description", "-"),
            flat_json.get("quantity", "-"),
            "EA",
            hu_number,
            flat_json.get("vendor", "UNKNOWN"),
            "",
            flat_json.get("vendor_lot", "-"),
            msd_level,
            flat_json.get("date_code", "-")
        )

        flat_json["hu"] = hu_number
        flat_json["ibd"] = ibd_number
        flat_json["engine_used"] = engine_used or "fallback"
        flat_json["datamatrix"] = datamatrix_str
        flat_json["zpl"] = zpl_code
        flat_json["supplier_invoice"] = parsed_fields.get("supplier_invoice", "-")
        flat_json["msd_level"] = parsed_fields.get("msd_level", "-")
        flat_json["msd_date"] = parsed_fields.get("msd_date", "-")
        flat_json["barcode"] = parsed_fields.get("barcode", "-")

        try:
            saved_schema = save_generated_schema(
                flat_json.get("vendor", "UNKNOWN"),
                normalized_ocr_text,
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
            **flat_json,
            "engine_used": flat_json["engine_used"],
            "processing_time": f"{processing_time:.2f}s",
            "raw_text": flat_json.get("raw_text", ""),
            "image_hash": image_hash,
        }
        try:
            save_scan(record)
        except Exception as e:
            print(f"Failed to save scan: {e}")

        response_payload = build_response_payload(
            raw_text=normalized_ocr_text,
            blocks=ocr_blocks,
            parsed=parsed_fields,
            structured_data=flat_json,
            engine_used=engine_used or "fallback",
            confidence=confidence,
            processing_time=processing_time,
        )
        log(f"Response ready in {processing_time:.2f}s using {engine_used}")
        return response_payload
    except Exception as e:
        print(f"Unhandled scan error: {e}")
        return fallback_payload()
