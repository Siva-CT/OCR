import os
import logging
from typing import Tuple, Dict, Any
from google.cloud import vision
from google.auth.exceptions import DefaultCredentialsError
import cv2
from .base_ocr import BaseOCR

logger = logging.getLogger(__name__)

class GoogleVisionOCREngine(BaseOCR):
    def __init__(self):
        self.client = None
        
        cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        if cred_path:
            logger.info(f"Google Vision checking credentials at: {cred_path}")
        else:
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not set in environment.")

        # Attempt to load, if it fails, capture the exception so it doesn't crash on app startup
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Vision client initialized successfully.")
        except DefaultCredentialsError:
            logger.error("WARNING: Google Vision credentials not found or invalid. Engine will fail upon usage.")
        except Exception as e:
            logger.error(f"WARNING: Google Vision initialization failed: {e}")

    def extract_text_and_layout(self, image) -> Tuple[str, Dict[str, Any]]:
        """
        Extracts both the raw text string and layout-aware mapped dictionaries.
        Traverses blocks locally associating nearest elements cleanly bypassing generic regex fails.
        """
        if not self.client:
            raise RuntimeError("Google Vision client is not initialized.")

        success, encoded = cv2.imencode(".jpg", image)
        if not success:
            return "", {}

        content = encoded.tobytes()
        vision_image = vision.Image(content=content)

        layout_data = {}
        extracted_text = ""
        
        try:
            response = self.client.text_detection(image=vision_image)
            
            if response.error.message:
                raise Exception(f"Vision API Error: {response.error.message}")

            texts = response.text_annotations
            if texts and len(texts) > 0:
                extracted_text = texts[0].description
                print("--- GOOGLE VISION TEXT ---")
                print(extracted_text)
            
            # Layout parsing 
            full_annotation = response.full_text_annotation
            if full_annotation.pages:
                for page in full_annotation.pages:
                    for block in page.blocks:
                        block_text = ""
                        for paragraph in block.paragraphs:
                            for word in paragraph.words:
                                word_text = ''.join([symbol.text for symbol in word.symbols])
                                block_text += word_text + " "
                            block_text += "\n"
                        
                        block_text = block_text.strip()
                        if not block_text:
                            continue
                            
                        print(f"--- GOOGLE VISION BLOCK TEXT ---\n{block_text}\n-------------------")
                        
                        # Apply naive pattern matching locally per block bridging split geometries
                        import re
                        block_lower = block_text.lower()
                        
                        # Spatial Proximity matches assuming keys and values fell into the same block boundary
                        
                        # IBD
                        if "ibd" in block_lower:
                            match = re.search(r"IBD.*?(\d+)", block_text, re.IGNORECASE)
                            if match: layout_data['ibd_no'] = match.group(1)
                            
                        # Part
                        if "part" in block_lower:
                            match = re.search(r"Part.*?([A-Z0-9][A-Z0-9\-]+)", block_text, re.IGNORECASE)
                            if match: layout_data['part_number'] = match.group(1).lstrip('-')
                            
                        # Vendor
                        if "vendor" in block_lower:
                            match = re.search(r"Vendor.*?(\d+)\s*[/|\s]\s*(.+)", block_text, re.IGNORECASE)
                            if match:
                                layout_data['vendor_id'] = match.group(1).strip()
                                layout_data['vendor_name'] = match.group(2).strip()
                            else:
                                match2 = re.search(r"Vendor.*?(\d+)", block_text, re.IGNORECASE)
                                if match2: layout_data['vendor_id'] = match2.group(1).strip()
                                
                        # Invoice
                        if "invoice" in block_lower:
                            match = re.search(r"Invoice.*?(\d+)", block_text, re.IGNORECASE)
                            if match: layout_data['supplier_invoice_number'] = match.group(1).strip()
                            
                            d_match = re.search(r"\d{2}[\./]\d{2}[\./]\d{4}", block_text)
                            if d_match: layout_data['supplier_invoice_date'] = d_match.group(0)
                        
                        # Lot
                        if "lot" in block_lower:
                            match = re.search(r"Lot.*?([A-Z0-9\-\.]+)", block_text, re.IGNORECASE)
                            if match: layout_data['vendor_lot_number'] = match.group(1).strip()
                        
                        # MSD
                        if "msd" in block_lower:
                            l_match = re.search(r"Level.*?(\d+)", block_text, re.IGNORECASE)
                            if l_match: layout_data['msd_level'] = l_match.group(1).strip()
                            d_match = re.search(r"Date.*?(\d+)", block_text, re.IGNORECASE)
                            if d_match: layout_data['msd_date'] = d_match.group(1).strip()

        except Exception as e:
            print(f"ERROR: Google Vision extraction failed: {e}")
            raise e

        return extracted_text, layout_data

    def extract_text(self, image) -> str:
        # Fallback ensuring Base abstractions don't break
        text, _ = self.extract_text_and_layout(image)
        return text