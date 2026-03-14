from enum import Enum
from typing import List, Optional
from pydantic import BaseModel


class OCREngine(str, Enum):
    tesseract = "tesseract"
    easy_ocr = "easy_ocr"
    paddle_ocr = "paddle_ocr"
    google_vision = "google_vision"


class LabelData(BaseModel):
    ibd_no: Optional[str] = None
    part_number: Optional[str] = None
    description: Optional[str] = None
    quantity_per_pack: Optional[str] = None
    total_quantity: Optional[str] = None
    serial_number: Optional[str] = None
    vendor_id: Optional[str] = None
    vendor_name: Optional[str] = None
    supplier_invoice_number: Optional[str] = None
    supplier_invoice_date: Optional[str] = None
    vendor_lot_number: Optional[str] = None
    msd_level: Optional[str] = None
    msd_date: Optional[str] = None
    raw_text: Optional[str] = None


class ProcessLabelResponse(BaseModel):
    engine_used: str
    labels: List[LabelData]
