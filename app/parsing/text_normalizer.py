import re

def normalize_ocr_text(text: str) -> str:
    """
    Cleans common OCR misreads and normalizes punctuation/spacing before regex parsing.
    Shared module invoked across all OCR engines natively.
    """
    if not text:
        return text

    # Common substitution loops
    replacements = {
        "€": "E",
        "®": "0",
        "|": "1",
        "BA": "EA",
        "endor": "Vendor",
        "VVendor": "Vendor",
        "jupplier": "Supplier",
        "Lot Na": "Lot No",
        "Von": "Ven",
        "T0252": "TO252",
        "QTY :": "QTY:",
        "O.1": "0.1",
        "AVNETASIAPTE": "AVNET ASIA PTE",
        "AVNST": "AVNET",
        "PTR": "PTE"
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        
    # Sometimes OCR reads 'O' instead of '0' in numbers.
    text = re.sub(r'(?<=\d)[Oo]', '0', text)
    text = re.sub(r'[Oo](?=\d)', '0', text)

    # Force uniform multi-spaces scaling OCR spacing discrepancies natively
    text = re.sub(r"\s+", " ", text)
    
    return text
