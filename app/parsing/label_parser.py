import re

from .text_normalizer import normalize_ocr_text  # noqa: F401 - re-exported for pipeline.py backward compat


def parse_label_text(text: str) -> dict:
    """
    Parses OCR text into structured label fields using OCR-tolerant regex.
    Missing fields are returned as "-".
    """

    def match(pattern: str, default: str = "-") -> str:
        found = re.search(pattern, normalized, re.IGNORECASE)
        if not found:
            return default
        value = found.group(found.lastindex or 0).strip()
        return value or default

    if not text:
        return {
            "ibd_no": "-",
            "part_number": "-",
            "description": "-",
            "quantity_per_pack": "-",
            "total_quantity": "-",
            "serial_number": "-",
            "vendor_id": "-",
            "vendor_name": "-",
            "supplier_invoice_number": "-",
            "supplier_invoice_date": "-",
            "vendor_lot_number": "-",
            "msd_level": "-",
            "msd_date": "-",
            "raw_text": "-",
        }

    normalized = normalize_ocr_text(text)
    vendor_name = "-"
    for candidate in ["PANJIT", "LITTELFUSE", "KEMET", "BOURNS", "YAGEO", "DIODES", "ACME", "NXP"]:
        if candidate in normalized:
            vendor_name = candidate
            break

    return {
        "ibd_no": match(r"IBD\s*NO[#:]?\s*(\w+)"),
        "part_number": match(r"(?:PART(?:\s*(?:NO|NUMBER|NUMGER))?|MPN|P/?N|PIN)\s*[:#.\-]?\s*([A-Z0-9][A-Z0-9\-_/.]+)"),
        "description": normalized[:160] if normalized else "-",
        "quantity_per_pack": match(r"(?:QTY|QUANTITY|Q)\s*[:\-]?\s*(\d+)"),
        "total_quantity": match(r"TOTAL\s*(?:QTY|QUANTITY)\s*[:\-]?\s*(\d+)"),  # Distinguish from serial
        "serial_number": match(r"SERIAL\s*(?:NO|NUMBER)\s*[:\-]?\s*(\d{15,20})"),  # Distinguish from quantity
        "vendor_id": match(r"VENDOR\s*[:\-]?\s*(\d+)"),
        "vendor_name": vendor_name,
        "supplier_invoice_number": match(r"SUPPLIER\s*INVOICE\s*[:\-]?\s*(\S+)"),
        "supplier_invoice_date": match(r"\d{2}[\./]\d{2}[\./]\d{4}"),
        "vendor_lot_number": match(r"VEN\s*LOT\s*NO\s*[:\-]?\s*(\S+)"),
        "msd_level": match(r"MSD\s*LEVEL\s*[:\-]?\s*(\S+)"),
        "msd_date": match(r"MSD\s*DATE\s*[:\-]?\s*(\S+)"),
        "raw_text": normalized or "-",
    }
