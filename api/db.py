import os
import hashlib
from datetime import datetime
import pandas as pd

DATA_DIR = "/app/data"
LOGS_DIR = "/app/logs"
SCANS_CSV = os.path.join(DATA_DIR, "scans.csv")
SCANS_EXCEL = os.path.join(DATA_DIR, "scans.xlsx")


def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(LOGS_DIR, exist_ok=True)


def save_scan(record: dict):
    """
    Persist a single scan into CSV and Excel for auditability.
    Required keys in record: vendor, part_number, quantity, vendor_lot, date_code, description,
    engine_used, processing_time, raw_text, image_hash, hu, ibd.
    """
    ensure_dirs()
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vendor": record.get("vendor", ""),
        "part_number": record.get("part_number", ""),
        "quantity": record.get("quantity", ""),
        "vendor_lot": record.get("vendor_lot", ""),
        "date_code": record.get("date_code", ""),
        "description": record.get("description", ""),
        "engine": record.get("engine_used", ""),
        "processing_time": record.get("processing_time", ""),
        "raw_text": record.get("raw_text", ""),
        "image_hash": record.get("image_hash", ""),
        "hu": record.get("hu", ""),
        "ibd": record.get("ibd", ""),
    }
    df = pd.DataFrame([row])

    # CSV
    try:
        if os.path.exists(SCANS_CSV):
            existing = pd.read_csv(SCANS_CSV)
            pd.concat([existing, df], ignore_index=True).to_csv(SCANS_CSV, index=False)
        else:
            df.to_csv(SCANS_CSV, index=False)
    except Exception as e:
        print(f"CSV write failed: {e}")

    # Excel
    try:
        if os.path.exists(SCANS_EXCEL):
            existing = pd.read_excel(SCANS_EXCEL)
            pd.concat([existing, df], ignore_index=True).to_excel(SCANS_EXCEL, index=False)
        else:
            df.to_excel(SCANS_EXCEL, index=False)
    except Exception as e:
        print(f"Excel write failed: {e}")


def is_duplicate(part_number: str, vendor_lot: str) -> bool:
    """
    Fast duplicate check based on hash of part_number + vendor_lot.
    """
    if not os.path.exists(SCANS_CSV):
        return False
    try:
        if not part_number or not vendor_lot:
            return False
        incoming_hash = hashlib.sha256(f"{part_number}{vendor_lot}".encode()).hexdigest()
        df = pd.read_csv(SCANS_CSV)
        if 'part_number' not in df.columns or 'vendor_lot' not in df.columns:
            return False
        df["derived_hash"] = df.apply(
            lambda row: hashlib.sha256(f"{str(row['part_number'])}{str(row['vendor_lot'])}".encode()).hexdigest(),
            axis=1,
        )
        return incoming_hash in df["derived_hash"].values
    except Exception as e:
        print(f"Duplicate check failed: {e}")
        return False
