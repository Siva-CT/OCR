from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import vision
import os

app = FastAPI(title="Google Vision Service")


def _empty_response():
    return {
        "raw_text": "",
        "blocks": [],
    }


def _annotation_box(annotation) -> list[list[int]]:
    vertices = []
    bounding_poly = getattr(annotation, "bounding_poly", None)
    for vertex in getattr(bounding_poly, "vertices", []) or []:
        vertices.append([int(getattr(vertex, "x", 0) or 0), int(getattr(vertex, "y", 0) or 0)])
    return vertices


def _resolve_credentials_path() -> str:
    candidates = [
        os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
        "/app/keys/google-vision-key.json",
        "/app/keys/google_vision_key.json",
    ]

    for path in candidates:
        if path and os.path.exists(path):
            return path
    return ""


def _build_ocr_response(response) -> dict:
    annotations = list(getattr(response, "text_annotations", []) or [])
    if not annotations:
        return _empty_response()

    raw_text = str(getattr(annotations[0], "description", "") or "").strip()
    blocks = []

    for annotation in annotations[1:]:
        text = str(getattr(annotation, "description", "") or "").strip()
        if not text:
            continue
        blocks.append({
            "text": text,
            "confidence": 1.0,
            "box": _annotation_box(annotation),
        })

    if not raw_text:
        raw_text = "\n".join(block["text"] for block in blocks)

    return {
        "raw_text": raw_text,
        "blocks": blocks,
    }


# Note: GOOGLE_APPLICATION_CREDENTIALS environment variable must be set
# when running this service for it to authenticate with GCP.
@app.post("/ocr")
async def process_image(file: UploadFile = File(...)):
    if not str(file.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    contents = await file.read()
    
    credentials_path = _resolve_credentials_path()

    # If credentials aren't available, return a mock response for offline dev.
    if not credentials_path:
        # Mock mode mapping for testing locally without billing enabled.
        print("WARNING: Using mock response - Google Vision credentials missing", flush=True)
        return {
            "raw_text": "MOCK TEXT DATA 1P E28-01941-0258-A Q 2500 1T KEM-22B-99",
            "blocks": [
                {"text": "MOCK", "confidence": 1.0, "box": [[0,0],[10,0],[10,10],[0,10]]}
            ]
        }

    try:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
        client = vision.ImageAnnotatorClient()
        image = vision.Image(content=contents)
        response = client.text_detection(image=image)
        
        if response.error.message:
            raise Exception(response.error.message)

        return _build_ocr_response(response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google Vision API error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy", "engine": "google_vision"}
