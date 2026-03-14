import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize only once (safe for FastAPI startup with multiple workers)
try:
    app = firebase_admin.get_app()
except ValueError:
    key_path = os.environ.get("FIREBASE_KEY_PATH", "firebase-key.json")
    if os.path.exists(key_path):
        cred = credentials.Certificate(key_path)
        firebase_admin.initialize_app(cred)
    else:
        print(f"WARNING: Firebase key not found at {key_path}")

db = firestore.client()


def save_label(data: dict) -> str:
    """
    Saves a parsed label dict to Firestore 'labels' collection.
    Uses serial_number as the document ID to prevent duplicates on re-scan.
    Falls back to auto-generated ID when serial is absent.
    Returns the document ID used.
    """
    try:
        db = firestore.client()
    except ValueError:
        print("[Firebase] Not initialized")
        return ""
    
    record = dict(data)
    record["timestamp"] = datetime.utcnow().isoformat()
    # Remove None values — Firestore doesn't store nulls cleanly
    record = {k: v for k, v in record.items() if v is not None}

    serial = record.get("serial_number")
    print(f"[Firebase] Saving label: {serial}")

    if serial:
        # Use serial number as doc ID — idempotent on re-scans
        db.collection("labels").document(str(serial)).set(record)
        return str(serial)
    else:
        # No serial — auto-generate a unique doc ID
        _, doc_ref = db.collection("labels").add(record)
        print(f"[Firebase] Auto-ID assigned: {doc_ref.id}")
        return doc_ref.id

