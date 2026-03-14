import json
import re
import time
import subprocess
from typing import Dict, List

KEMET_TEMPLATE = {
    "ibd_no": "185101626",
    "part": "E02-00463-0104-A",
    "description_lines": [
        "CAPACITOR,0.1UF/50,0805,FLOATING",
        "ELECTR",
        "DOL",
    ],
    "vendor_code": "900100576",
    "vendor_display": "AVNET ASIA PTE",
    "supplier_invoice": "1182614732/21.01.2026",
    "msd_level": "1",
    "msd_date": "00000000",
    "reference_number": "100000100018320103",
    "pack_qty": "1500000",
}

JAUCH_TEMPLATE = {
    "ibd_no": "185099607",
    "part": "E26-05166-S53P-A",
    "description_lines": [
        "CRYSTAL 8MHz JAUCH",
        "JXS53P4-10-20/20-T1",
    ],
    "vendor_code": "900407545",
    "vendor_display": "PERFECT SALES E",
    "supplier_invoice": "WHM001/252604102/31.01.2026",
    "msd_level": "1",
    "msd_date": "00000000",
    "reference_number": "100000100018328942",
    "pack_qty": "30000",
}

TEMPLATES = {
    "KEMET": KEMET_TEMPLATE,
    "JAUCH": JAUCH_TEMPLATE,
}

DEFAULT_TEMPLATE_VENDOR = "KEMET"


def generate_hu_number() -> str:
    prefix = "10000010"
    timestamp_part = str(int(time.time() * 100))[-10:]
    hu = f"{prefix}{timestamp_part}"
    result = hu[:18].ljust(18, "0")
    assert len(result) == 18, f"Generated HU should be 18 chars, got {len(result)}"
    return result


def generate_ibd_number() -> str:
    prefix = "IBD"
    timestamp_part = str(int(time.time() * 1000))[-10:]
    return f"{prefix}{timestamp_part}"


def generate_datamatrix_string(hu, part, vendor_lot, scanned_qty, msd_level, vendor_code) -> str:
    return f"RID{hu}PRN{part}LOT{vendor_lot}QTY{scanned_qty}MSD{msd_level}VEN{vendor_code}"


def _normalize_quantity(value: object) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value)
    text = str(value).strip()
    if not text:
        return "-"
    if re.fullmatch(r"\d+\.0+", text):
        return text.split(".")[0]
    return text


def detect_vendor_template(ocr_vendor_name: str) -> str:
    normalized = str(ocr_vendor_name or "").upper()
    if "KEMET" in normalized:
        return "KEMET"
    if "JAUCH" in normalized:
        return "JAUCH"
    return "UNKNOWN"


def build_minda_label_fields(ocr_vendor_name: str, quantity: object, vendor_lot: object) -> Dict[str, object]:
    detected_vendor = detect_vendor_template(ocr_vendor_name)
    
    if detected_vendor == "UNKNOWN":
        return {
            "template_vendor": "UNKNOWN",
            "ibd_no": "",
            "part": "",
            "description_lines": [],
            "description": "",
            "qty": "",
            "qty_display": "",
            "pack_qty": "",
            "vendor_code": "",
            "vendor_display": "",
            "vendor_lot": "",
            "supplier_invoice": "",
            "msd_level": "",
            "msd_date": "",
            "reference_number": "",
        }

    template_vendor = detected_vendor if detected_vendor in TEMPLATES else DEFAULT_TEMPLATE_VENDOR
    template = TEMPLATES[template_vendor]

    qty_value = _normalize_quantity(quantity)
    vendor_lot_value = str(vendor_lot or "-").strip() or "-"
    description_lines = list(template["description_lines"])
    description_payload = "\n".join([line for line in description_lines if line]).strip() or "-"
    pack_qty = str(template.get("pack_qty", "-") or "-")
    qty_display = f"{qty_value} /{pack_qty} EA" if qty_value != "-" and pack_qty != "-" else qty_value

    return {
        "template_vendor": template_vendor,
        "ibd_no": template["ibd_no"],
        "part": template["part"],
        "description_lines": description_lines,
        "description": description_payload,
        "qty": qty_value,
        "qty_display": qty_display,
        "pack_qty": pack_qty,
        "vendor_code": template["vendor_code"],
        "vendor_display": template["vendor_display"],
        "vendor_lot": vendor_lot_value,
        "supplier_invoice": template["supplier_invoice"],
        "msd_level": template["msd_level"],
        "msd_date": template["msd_date"],
        "reference_number": template["reference_number"],
    }


def generate_qr_payload(label_fields: Dict[str, object]) -> str:
    unique_number = label_fields.get("reference_number") or label_fields.get("barcode_number") or ""
    part_number = label_fields.get("part") or ""
    lot_number = label_fields.get("vendor_lot") or ""
    quantity = label_fields.get("qty") or ""
    msd_number = label_fields.get("msd_level") or ""
    vendor_code = label_fields.get("vendor_code") or ""
    return generate_datamatrix_string(
        str(unique_number),
        str(part_number),
        str(lot_number),
        str(quantity),
        str(msd_number),
        str(vendor_code),
    )


def generate_zpl(qr_payload: str, label_fields: Dict[str, object], barcode_number: str) -> str:
    description_lines: List[str] = list(label_fields.get("description_lines", []))
    desc_line_1 = description_lines[0] if len(description_lines) > 0 else "-"
    desc_line_2 = description_lines[1] if len(description_lines) > 1 else ""
    desc_line_3 = description_lines[2] if len(description_lines) > 2 else ""
    qty_display = label_fields.get("qty_display", label_fields.get("qty", "-"))

    zpl = f"""^XA
^FX QR Code
^FO50,50^BQN,2,5
^FDLA,{qr_payload}^FS

^FX Text Info Right Side
^FO350,50^A0N,30,30^FDIBD No : {label_fields.get('ibd_no', '-')}^FS
^FO350,90^A0N,30,30^FDPart : {label_fields.get('part', '-')}^FS
^FO350,130^A0N,30,30^FD{desc_line_1}^FS
^FO350,155^A0N,30,30^FD{desc_line_2}^FS
^FO350,180^A0N,30,30^FD{desc_line_3}^FS
^FO350,205^A0N,30,30^FDQTY:{qty_display}^FS

^FX Divider
^FO50,230^GB700,1,1^FS

^FX Bottom Info
^FO50,245^A0N,40,40^FB700,1,0,L^FD{barcode_number}^FS
^FO50,285^A0N,30,30^FDVendor : {label_fields.get('vendor_code', '-')} / {label_fields.get('vendor_display', '-')}^FS
^FO50,315^A0N,30,30^FDSupplier Invoice : {label_fields.get('supplier_invoice', '-')}^FS
^FO50,345^A0N,30,30^FDVen Lot No : {label_fields.get('vendor_lot', '-')}^FS
^FO50,375^A0N,30,30^FDMSD Level : {label_fields.get('msd_level', '-')}   MSD Date : {label_fields.get('msd_date', '-')}^FS
^XZ"""
    return zpl


def print_label(zpl_code):
    import platform
    try:
        with open("/tmp/label.zpl", "w") as f:
            f.write(zpl_code)

        if platform.system() == "Windows":
            # Windows print command
            subprocess.run([
                "powershell", "-Command",
                f'Add-PrinterPort -Name "FILE:" -PrinterPortType File; Print-Object -FilePath /tmp/label.zpl -PrinterName "zebra_printer"'
            ], check=False)
        else:
            # Unix/Linux print command
            subprocess.run([
                "lp",
                "-d",
                "zebra_printer",
                "/tmp/label.zpl"
            ], check=True)
        return True
    except Exception as e:
        print(f"Print failed: {e}")
        return False

