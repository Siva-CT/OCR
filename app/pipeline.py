import re
import cv2
import numpy as np

from .schemas import LabelData, ProcessLabelResponse
from .preprocessing.image_cleaner import preprocess_image
from .ocr.ocr_manager import get_ocr_engine
from .parsing.label_parser import parse_label_text
from .detection.label_detector import detect_labels
from .detection.yolo_detector import detect_labels_yolo


def process_label_image(image_bytes, engine) -> ProcessLabelResponse:
    """
    Orchestrates the entire label processing pipeline.
    Handles multiple labels in one image.
    """
    # Convert bytes to numpy image
    np_img = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Invalid image or failed to decode")

    # Detect distinct label bounding boxes
    try:
        cropped_boxes = detect_labels_yolo(image)
        if not cropped_boxes:
            raise Exception("YOLO mapped 0 regions")
    except Exception as e:
        print(f"YOLO detection failed or missing: {e}. Falling back to OpenCV.")
        cropped_boxes = detect_labels(image)

    # Initialize results
    labels_list = []

    ocr_engine = get_ocr_engine(engine)
    engine_name = engine.value if hasattr(engine, "value") else str(engine).lower()

    print(f"Detected label regions: {len(cropped_boxes)}")

    # Process each detected distinct label block
    for i, box_or_crop in enumerate(cropped_boxes):
        print(f"Processing label region {i + 1}")
        
        if isinstance(box_or_crop, tuple):
            x, y, w, h = box_or_crop
            # Apply 30px padding natively for Coordinate maps
            pad = 30
            img_h, img_w = image.shape[:2]
            px1 = max(0, x - pad)
            py1 = max(0, y - pad)
            px2 = min(img_w, x + w + pad)
            py2 = min(img_h, y + h + pad)
            cropped = image[py1:py2, px1:px2]
        else:
            cropped = box_or_crop

        # Preprocess localized image
        processed_img = preprocess_image(cropped)

        try:
            layout_data = {}
            if hasattr(ocr_engine, "extract_text_and_layout"):
                text, layout_data = ocr_engine.extract_text_and_layout(processed_img)
            else:
                text = ocr_engine.extract_text(processed_img)

            print(f"--- OCR RAW TEXT [{engine_name}] ---")
            print(text)
            print("---------------------------------------")

            from .parsing.label_parser import normalize_ocr_text
            normalized_text = normalize_ocr_text(text)

            print(f"--- NORMALIZED TEXT [{engine_name}] ---")
            print(normalized_text)
            print("---------------------------------------")

            label_blocks = split_labels_by_serial(normalized_text)
            print(f"[serial-split] Found {len(label_blocks)} label block(s) in OCR text")

            for block_idx, block in enumerate(label_blocks):
                block_parsed = parse_label_text(block)

                if block_idx == 0 and layout_data:
                    print(f"--- LAYOUT EXTRACTED FIELDS [{engine_name}] ---")
                    print(layout_data)
                    print("---------------------------------------")
                    for k, v in layout_data.items():
                        if v and block_parsed.get(k, "-") in {"", "-"}:
                            block_parsed[k] = v

                block_parsed["raw_text"] = block or "-"

                print(f"--- PARSED FIELDS [{engine_name}] block {block_idx + 1} ---")
                print(block_parsed)
                print("---------------------------------------")

                _save_to_firebase(block_parsed)
                labels_list.append(LabelData(**block_parsed))

        except Exception as e:
            print(f"OCR Engine [{engine_name}] Failed on segment: {e}")
            fallback_label = {"raw_text": "-", "vendor_name": "-", "part_number": "-"}
            _save_to_firebase(fallback_label)
            labels_list.append(LabelData(**fallback_label))

    return ProcessLabelResponse(
        engine_used=engine_name,
        labels=labels_list,
    )


def split_labels_by_serial(text: str) -> list:
    """
    Splits OCR text into individual label blocks using 17-digit serial numbers
    as natural delimiters. Returns the original text as a single block if no
    serials are detected.
    """
    serials = re.findall(r"\b\d{17}\b", text)
    if not serials:
        return [text]

    parts = re.split(r"\b\d{17}\b", text)
    blocks = []
    for idx, serial in enumerate(serials):
        # Attach the text before this serial plus the serial itself
        block = (parts[idx] if idx < len(parts) else "") + serial
        blocks.append(block.strip())

    return blocks


def _save_to_firebase(data: dict):
    """
    Attempts to persist label data to Firestore.
    Wrapped in try/except so Firestore errors never abort the OCR response.
    """
    try:
        from .services.firebase_service import save_label
        save_label(data)
    except Exception as e:
        print(f"[Firebase] Save skipped: {e}")
