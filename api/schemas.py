import difflib
import json
import os
import re
import time
from typing import Any, Dict, List

SCHEMA_DIR = os.environ.get("SCHEMA_DIR", "/app/schemas")

DEFAULT_SCHEMA_PATTERNS = {
    "ibd_no": r"IBD\s*NO[#:]?\s*(\w+)",
    "barcode": r"\b\d{15,20}\b",
    "supplier_invoice": r"SUPPLIER\s*INVOICE\s*[:\-]?\s*(\S+)",
    "ven_lot_no": r"VEN\s*LOT\s*NO\s*[:\-]?\s*(\S+)",
    "msd_level": r"MSD\s*LEVEL\s*[:\-]?\s*(\S+)",
    "msd_date": r"MSD\s*DATE\s*[:\-]?\s*(\S+)",
}

DEFAULT_PATTERNS = {
    "part_number": r"(?:PART(?:\s*(?:NO|NUMBER|NUMGER))?|MPN|P/?N|PIN|KEMET\s*P/?N)\s*[:#.\-]?\s*([A-Z0-9-]{6,25})",
    "quantity": r"(?:QTY|QUANTITY|GTY|OTY)\s*[:#.\-]?\s*(\d{1,8})",
    "vendor_lot": r"(?:VEN\s*LOT\s*NO|LOT\s*NUMBER|LOT\s*NO|LOTNO|LOT)\s*[:#.\-]?\s*([A-Z0-9-]{4,25})",
    "date_code": r"(?:DATE\s*CODE|DATECODE|DC|D/C|DIC|DATE)\s*[:#.\-]?\s*(\d{2,8})",
}

DEFAULT_KEYWORDS = {
    "part_number": ["PART NUMBER", "PART NUMGER", "PART NO", "PART", "MPN", "P/N", "PN", "PIN"],
    "quantity": ["QUANTITY", "QTY", "Q", "GTY", "OTY"],
    "vendor_lot": ["VEN LOT NO", "LOT NUMBER", "LOT NO", "LOT"],
    "date_code": ["DATE CODE", "DATE", "DC", "D/C", "DIC"],
}

VENDOR_KEYWORDS = [
    "KEMET",
    "PANJIT",
    "LITTELFUSE",
    "BOURNS",
    "KEMET A YAGEO COMPANY",
    "YAGEO COMPANY",
    "YAGEO",
    "DIODES",
]

NOISE_WORDS = {
    "CAPACITOR",
    "ROHS",
    "PRC",
    "OF",
    "MEXICO",
    "UF",
    "V",
}

FILLER_TOKENS = {
    "PART",
    "PARTNO",
    "PARTNUMBER",
    "PARTNUMGER",
    "NUMBER",
    "NO",
    "MPN",
    "PN",
    "PIN",
    "LOT",
    "LOTNO",
    "LOTNUMBER",
    "QTY",
    "QUANTITY",
    "Q",
    "DATE",
    "DATECODE",
    "DC",
    "CODE",
    "VEN",
}

PART_VALUE_PATTERN = re.compile(r"^[A-Z0-9-]{6,25}$")
QUANTITY_VALUE_PATTERN = re.compile(r"\b\d{1,8}\b")
LOT_VALUE_PATTERN = re.compile(r"^[A-Z0-9-]{4,25}$")
DATE_VALUE_PATTERN = re.compile(r"\b\d{2,8}\b")
BARCODE_PATTERN = re.compile(r"^\d{15,20}$")
IBD_VALUE_PATTERN = re.compile(r"^(?:IBD)?[A-Z0-9]{6,20}$")

PACKAGE_PREFIX_FIXUPS = {
    "CO805": "C0805",
    "CO603": "C0603",
    "CO402": "C0402",
    "CO1206": "C1206",
}

KNOWN_PART_CORRECTIONS = {
    "BAV70W-AU": "BAV99-AU",
}

SCHEMA_SCORE_FIELDS = (
    "ibd_no",
    "barcode",
    "part",
    "qty",
    "supplier_invoice",
    "ven_lot_no",
    "msd_level",
    "msd_date",
)


def log(msg: str):
    print(f"[SCHEMA] {msg}", flush=True)


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _write_json(path: str, payload: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _schema_file_name(vendor_name: str) -> str:
    normalized_vendor = re.sub(r"[^A-Z0-9]+", "_", str(vendor_name or "").strip().upper()).strip("_")
    if not normalized_vendor:
        normalized_vendor = "UNKNOWN"
    return f"{normalized_vendor.lower()}.json"


def _schema_path(vendor_name: str) -> str:
    return os.path.join(SCHEMA_DIR, _schema_file_name(vendor_name))


def normalize_ocr_text(text: str) -> str:
    normalized = (text or "").upper()
    normalized = normalized.replace("\r", "\n")
    normalized = normalized.replace("\t", " ")
    normalized = re.sub(r"(?<=\d)O(?=\d)|(?<=\b)O(?=\d)|(?<=\d)O(?=\b)", "0", normalized)
    normalized = re.sub(r"(?<=\d)I(?=\d)|(?<=\b)I(?=\d)|(?<=\d)I(?=\b)", "1", normalized)
    normalized = re.sub(r"\bPARTNUMBER\b", "PART NUMBER", normalized)
    normalized = re.sub(r"\bPARTNO\b", "PART NO", normalized)
    normalized = re.sub(r"\bLOTNUMBER\b", "LOT NUMBER", normalized)
    normalized = re.sub(r"\bLOTNO\b", "LOT NO", normalized)
    normalized = re.sub(r"\bDATECODE\b", "DATE CODE", normalized)
    normalized = re.sub(r"\bSUPPLIERINVOICE\b", "SUPPLIER INVOICE", normalized)
    normalized = re.sub(r"\bVENLOTNO\b", "VEN LOT NO", normalized)
    normalized = re.sub(r"\bMSDLEVEL\b", "MSD LEVEL", normalized)
    normalized = re.sub(r"\bMSDDATE\b", "MSD DATE", normalized)
    normalized = re.sub(r"\b(?:GTY|OTY)\b", "QTY", normalized)
    normalized = re.sub(r"\bDIC\b", "DC", normalized)
    normalized = re.sub(r"\b0C\b", "DC", normalized)
    normalized = re.sub(r"PART\s+[A-Z]\s+NO", "PART NO", normalized)
    normalized = re.sub(r"LOT\s+[A-Z]\s+NO", "LOT NO", normalized)
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n\s+", "\n", normalized)
    normalized = re.sub(r"\n+", "\n", normalized)
    return normalized.strip()


def _normalize_lines(text: str) -> List[str]:
    return [line.strip() for line in normalize_ocr_text(text).split("\n") if line.strip()]


def detect_vendor_from_text(text: str, vendor_hint: str = "") -> str:
    if vendor_hint and str(vendor_hint).strip().upper() not in {"", "-", "UNKNOWN"}:
        return str(vendor_hint).strip().upper()

    normalized = normalize_ocr_text(text)
    for vendor_name in VENDOR_KEYWORDS:
        if vendor_name in normalized:
            detected = "YAGEO" if "YAGEO" in vendor_name and "KEMET" not in vendor_name else vendor_name
            log(f"Vendor detected by text: {detected}")
            return detected

    for token in normalized.replace("\n", " ").split():
        matches = difflib.get_close_matches(token, VENDOR_KEYWORDS, n=1, cutoff=0.82)
        if matches:
            detected = "YAGEO" if "YAGEO" in matches[0] and "KEMET" not in matches[0] else matches[0]
            log(f"Vendor detected by fuzzy text match: {detected}")
            return detected

    return "UNKNOWN"


def _extract_pattern(text: str, pattern: str) -> str:
    try:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            group_index = match.lastindex or 1
            value = match.group(group_index).strip()
            if value:
                log(f"Regex match: {value}")
                return value
    except Exception as exc:
        log(f"Regex error: {exc}")
    return ""


def extract_fields_from_text(text: str) -> Dict[str, str]:
    normalized = normalize_ocr_text(text)
    return {
        "part_number": _extract_pattern(normalized, DEFAULT_PATTERNS["part_number"]),
        "quantity": _extract_pattern(normalized, DEFAULT_PATTERNS["quantity"]),
        "vendor_lot": _extract_pattern(normalized, DEFAULT_PATTERNS["vendor_lot"]),
        "date_code": _extract_pattern(normalized, DEFAULT_PATTERNS["date_code"]),
    }


def _extract_tail_value(lines: List[str], index: int, aliases: List[str]) -> str:
    def next_meaningful_line(start_index: int) -> str:
        for candidate_index in range(start_index + 1, min(len(lines), start_index + 4)):
            candidate_line = lines[candidate_index].strip()
            normalized_candidate = re.sub(r"[^A-Z0-9]", "", candidate_line.upper())
            if normalized_candidate in {"", "NO", "NUMBER", "NUMGER", "PART", "LOT", "DATE", "CODE", "DC"}:
                continue
            return candidate_line
        return ""

    line = lines[index]
    compact_line = re.sub(r"[^A-Z0-9]", "", line.upper())
    ordered_aliases = sorted([alias.upper() for alias in aliases], key=len, reverse=True)

    for alias in ordered_aliases:
        pattern = rf"{re.escape(alias)}\s*[:#.\-]?\s*(.*)"
        match = re.search(pattern, line, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value
            fallback_value = next_meaningful_line(index)
            if fallback_value:
                return fallback_value

        compact_alias = re.sub(r"[^A-Z0-9]", "", alias)
        prefix = compact_line[: len(compact_alias)]
        if compact_alias and prefix and difflib.SequenceMatcher(None, prefix, compact_alias).ratio() >= 0.72:
            compact_tail = compact_line[len(compact_alias):].strip()
            if compact_tail:
                return compact_tail
            fallback_value = next_meaningful_line(index)
            if fallback_value:
                return re.sub(r"[^A-Z0-9\-/.]", "", fallback_value.upper())

    return ""


def _clean_candidate_text(value: str) -> str:
    cleaned = normalize_ocr_text(value or "")
    cleaned = cleaned.replace(":", " ")
    cleaned = cleaned.replace("#", " ")
    cleaned = cleaned.replace("/", "")
    cleaned = cleaned.replace(".", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def _clean_tokens(value: str) -> List[str]:
    tokens = re.findall(r"[A-Z0-9-]+", _clean_candidate_text(value))
    return [token for token in tokens if token and token not in NOISE_WORDS]


def _clean_text_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9\-/.]", "", normalize_ocr_text(value or ""))


def _correct_part_number(value: str) -> str:
    candidate = re.sub(r"[^A-Z0-9\-]", "", (value or "").upper())
    if not candidate:
        return ""

    candidate = re.sub(r"(?<=\d)O(?=\d)", "0", candidate)
    candidate = re.sub(r"(?<=[A-Z])0(?=[A-Z])", "O", candidate)

    for wrong_prefix, correct_prefix in PACKAGE_PREFIX_FIXUPS.items():
        if wrong_prefix in candidate:
            candidate = candidate.replace(wrong_prefix, correct_prefix)

    candidate = KNOWN_PART_CORRECTIONS.get(candidate, candidate)
    return candidate if PART_VALUE_PATTERN.fullmatch(candidate) else ""


def _extract_part_value(value: str) -> str:
    for token in _clean_tokens(value):
        normalized = re.sub(r"^(?:PARTNUMBER|PARTNUMGER|PARTNO|PART|MPN|PN)+", "", token)
        if token in FILLER_TOKENS or not normalized:
            continue
        if normalized in VENDOR_KEYWORDS or normalized in NOISE_WORDS:
            continue
        if normalized.startswith("PIN") and len(normalized) > 6:
            normalized = normalized[3:]
        if PART_VALUE_PATTERN.fullmatch(normalized) and not normalized.isdigit():
            return normalized
    return ""


def _extract_quantity_value(value: str) -> str:
    cleaned = _clean_candidate_text(value)
    match = re.search(r"(?:QTY|QUANTITY|Q)\s*[:\-]?\s*(\d+)", cleaned)
    if match:
        return match.group(1)
    match = QUANTITY_VALUE_PATTERN.search(cleaned)
    return match.group(0) if match else ""


def _extract_lot_value(value: str) -> str:
    cleaned = _clean_candidate_text(value)
    match = re.search(r"(?:VEN\s*LOT\s*NO|LOT\s*NUMBER|LOT\s*NO|LOT)\s*[:#.\-]?\s*([A-Z0-9-]+)", cleaned)
    if match:
        candidate = match.group(1).strip()
        if LOT_VALUE_PATTERN.fullmatch(candidate) and re.search(r"\d", candidate):
            return candidate

    for token in _clean_tokens(value):
        normalized = re.sub(r"^(?:VENLOTNO|LOTNUMBER|LOTNO|LOT)+", "", token)
        if token in FILLER_TOKENS or not normalized:
            continue
        if LOT_VALUE_PATTERN.fullmatch(normalized) and re.search(r"\d", normalized):
            return normalized
    return ""


def _extract_date_value(value: str) -> str:
    cleaned = _clean_candidate_text(value)
    match = re.search(r"(?:DATE\s*CODE|DATE|DC)\s*[:\-]?\s*(\d{2,8})", cleaned)
    if match:
        return match.group(1)
    match = DATE_VALUE_PATTERN.search(cleaned)
    return match.group(0) if match else ""


def extract_fields_by_keywords(lines: List[str], keywords: Dict[str, List[str]] = None) -> Dict[str, str]:
    field_keywords = keywords or DEFAULT_KEYWORDS
    extracted = {
        "part_number": "",
        "quantity": "",
        "vendor_lot": "",
        "date_code": "",
    }

    for index in range(len(lines)):
        if not extracted["part_number"]:
            part_tail = _extract_tail_value(lines, index, field_keywords.get("part_number", []))
            part_value = _correct_part_number(_extract_part_value(part_tail))
            if part_value:
                extracted["part_number"] = part_value
                log(f"Line match part_number: {part_value}")

        if not extracted["quantity"]:
            quantity_tail = _extract_tail_value(lines, index, field_keywords.get("quantity", []))
            quantity_value = _extract_quantity_value(quantity_tail)
            if quantity_value:
                extracted["quantity"] = quantity_value
                log(f"Line match quantity: {quantity_value}")

        if not extracted["vendor_lot"]:
            lot_tail = _extract_tail_value(lines, index, field_keywords.get("vendor_lot", []))
            lot_value = _extract_lot_value(lot_tail)
            if lot_value:
                extracted["vendor_lot"] = lot_value
                log(f"Line match vendor_lot: {lot_value}")

        if not extracted["date_code"]:
            date_tail = _extract_tail_value(lines, index, field_keywords.get("date_code", []))
            date_value = _extract_date_value(date_tail)
            if date_value:
                extracted["date_code"] = date_value
                log(f"Line match date_code: {date_value}")

    if not extracted["date_code"]:
        for line in lines:
            date_value = _extract_date_value(line)
            if date_value and date_value != extracted["quantity"]:
                extracted["date_code"] = date_value
                log(f"Fallback date_code: {date_value}")
                break

    return extracted


def _extract_auxiliary_fields(text: str) -> Dict[str, str]:
    normalized = normalize_ocr_text(text)
    extracted: Dict[str, str] = {}
    for field_name, pattern in DEFAULT_SCHEMA_PATTERNS.items():
        extracted[field_name] = _extract_pattern(normalized, pattern)
    return extracted


def _resolve_cleaned_value(field_name: str, *candidates: str) -> str:
    cleaners = {
        "part_number": lambda value: _correct_part_number(_extract_part_value(value)),
        "quantity": _extract_quantity_value,
        "vendor_lot": _extract_lot_value,
        "date_code": _extract_date_value,
    }

    cleaner = cleaners[field_name]
    for candidate in candidates:
        cleaned = cleaner(candidate)
        if cleaned:
            return cleaned
    return "-"


def _extract_label_fields(raw_text: str, vendor_hint: str = "") -> Dict[str, str]:
    normalized_text = normalize_ocr_text(raw_text)
    lines = _normalize_lines(raw_text)
    line_matches = extract_fields_by_keywords(lines)
    regex_matches = extract_fields_from_text(normalized_text)
    aux_matches = _extract_auxiliary_fields(normalized_text)
    vendor = detect_vendor_from_text(normalized_text, vendor_hint)

    part_number = _resolve_cleaned_value(
        "part_number",
        line_matches.get("part_number", ""),
        regex_matches.get("part_number", ""),
    )
    quantity = _resolve_cleaned_value(
        "quantity",
        line_matches.get("quantity", ""),
        regex_matches.get("quantity", ""),
    )
    vendor_lot = _resolve_cleaned_value(
        "vendor_lot",
        line_matches.get("vendor_lot", ""),
        regex_matches.get("vendor_lot", ""),
        aux_matches.get("ven_lot_no", ""),
    )
    date_code = _resolve_cleaned_value(
        "date_code",
        line_matches.get("date_code", ""),
        regex_matches.get("date_code", ""),
    )

    barcode = _clean_text_token(aux_matches.get("barcode", ""))
    if not BARCODE_PATTERN.fullmatch(barcode):
        barcode = "-"

    ibd_no = _clean_text_token(aux_matches.get("ibd_no", ""))
    if ibd_no and not IBD_VALUE_PATTERN.fullmatch(ibd_no):
        ibd_no = "-"
    elif ibd_no and not ibd_no.startswith("IBD"):
        ibd_no = f"IBD{ibd_no}"

    return {
        "vendor": vendor if vendor else "UNKNOWN",
        "part_number": part_number,
        "quantity": quantity,
        "vendor_lot": vendor_lot,
        "date_code": date_code,
        "supplier_invoice": _clean_text_token(aux_matches.get("supplier_invoice", "")) or "-",
        "msd_level": _clean_text_token(aux_matches.get("msd_level", "")) or "-",
        "msd_date": _clean_text_token(aux_matches.get("msd_date", "")) or "-",
        "ibd_no": ibd_no or "-",
        "barcode": barcode,
        "raw_text": normalized_text or "-",
    }


def parse_ocr_text_fields(raw_text: str, vendor_hint: str = "") -> Dict[str, str]:
    extracted = _extract_label_fields(raw_text, vendor_hint=vendor_hint)
    return {
        "ibd_no": extracted["ibd_no"],
        "barcode": extracted["barcode"],
        "part": extracted["part_number"],
        "qty": extracted["quantity"],
        "vendor": extracted["vendor"],
        "supplier_invoice": extracted["supplier_invoice"],
        "ven_lot_no": extracted["vendor_lot"],
        "msd_level": extracted["msd_level"],
        "msd_date": extracted["msd_date"],
    }


def map_extracted_text_to_json(raw_text: str, vendor_hint: str = "") -> Dict[str, str]:
    extracted = _extract_label_fields(raw_text, vendor_hint=vendor_hint)
    description = extracted["raw_text"][:160] if extracted["raw_text"] != "-" else "-"

    return {
        "vendor": extracted["vendor"],
        "part_number": extracted["part_number"],
        "quantity": extracted["quantity"],
        "vendor_lot": extracted["vendor_lot"],
        "description": description,
        "date_code": extracted["date_code"],
        "raw_text": extracted["raw_text"],
        "supplier_invoice": extracted["supplier_invoice"],
        "msd_level": extracted["msd_level"],
        "msd_date": extracted["msd_date"],
        "barcode": extracted["barcode"],
    }


def sanitize_structured_fields(data: Dict[str, Any]) -> Dict[str, Any]:
    sanitized = dict(data or {})

    sanitized["part_number"] = _resolve_cleaned_value("part_number", str(sanitized.get("part_number", "") or ""))
    sanitized["quantity"] = _resolve_cleaned_value("quantity", str(sanitized.get("quantity", "") or ""))
    sanitized["vendor_lot"] = _resolve_cleaned_value("vendor_lot", str(sanitized.get("vendor_lot", "") or ""))
    sanitized["date_code"] = _resolve_cleaned_value("date_code", str(sanitized.get("date_code", "") or ""))

    barcode_candidate = _clean_text_token(str(sanitized.get("barcode", "") or ""))
    sanitized["barcode"] = barcode_candidate if BARCODE_PATTERN.fullmatch(barcode_candidate) else "-"

    vendor = str(sanitized.get("vendor", "UNKNOWN") or "UNKNOWN").strip().upper()
    sanitized["vendor"] = vendor if vendor and vendor != "-" else "UNKNOWN"

    sanitized["description"] = str(sanitized.get("description", "-") or "-").strip() or "-"
    sanitized["raw_text"] = str(sanitized.get("raw_text", "-") or "-").strip() or "-"
    sanitized["supplier_invoice"] = _clean_text_token(str(sanitized.get("supplier_invoice", "-") or "-")) or "-"
    sanitized["msd_level"] = _clean_text_token(str(sanitized.get("msd_level", "-") or "-")) or "-"
    sanitized["msd_date"] = _clean_text_token(str(sanitized.get("msd_date", "-") or "-")) or "-"
    sanitized["ibd_no"] = _clean_text_token(str(sanitized.get("ibd_no", "-") or "-")) or "-"

    return sanitized


def _count_schema_matches(parsed: Dict[str, Any]) -> int:
    score = 0
    for field_name in SCHEMA_SCORE_FIELDS:
        value = str(parsed.get(field_name, "-") or "-").strip()
        if value not in {"", "-"}:
            score += 1

    vendor = str(parsed.get("vendor", "UNKNOWN") or "UNKNOWN").strip().upper()
    if vendor not in {"", "-", "UNKNOWN"}:
        score += 1
    return score


def _looks_like_successful_scan(raw_text: str, parsed: Dict[str, Any]) -> bool:
    return bool(normalize_ocr_text(raw_text)) and _count_schema_matches(parsed) > 0


def build_schema_payload(vendor_name: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    vendor = str(vendor_name or parsed.get("vendor") or "UNKNOWN").strip().upper() or "UNKNOWN"
    return {
        "vendor": vendor,
        "patterns": dict(DEFAULT_SCHEMA_PATTERNS),
        "match_score": _count_schema_matches(parsed),
        "updated_at": int(time.time()),
    }


def save_generated_schema(vendor_name: str, raw_text: str, parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not _looks_like_successful_scan(raw_text, parsed):
        return {}

    schema_payload = build_schema_payload(vendor_name, parsed)
    schema_path = _schema_path(schema_payload["vendor"])
    existing_schema = _read_json(schema_path)
    existing_score = int(existing_schema.get("match_score", 0) or 0)
    new_score = int(schema_payload.get("match_score", 0) or 0)

    if existing_schema and new_score <= existing_score:
        log(
            f"Keeping existing schema for {schema_payload['vendor']} "
            f"(existing score {existing_score}, new score {new_score})"
        )
        return existing_schema

    _write_json(schema_path, schema_payload)
    log(f"Saved schema for {schema_payload['vendor']} at {schema_path}")
    return schema_payload


def list_saved_schemas() -> List[Dict[str, Any]]:
    os.makedirs(SCHEMA_DIR, exist_ok=True)
    items: List[Dict[str, Any]] = []

    for file_name in sorted(os.listdir(SCHEMA_DIR)):
        if not file_name.endswith(".json"):
            continue

        payload = _read_json(os.path.join(SCHEMA_DIR, file_name))
        items.append({
            "vendor": str(payload.get("vendor", "UNKNOWN") or "UNKNOWN").upper(),
            "file": file_name,
            "match_score": int(payload.get("match_score", 0) or 0),
        })

    return items
