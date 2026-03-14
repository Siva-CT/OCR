import os
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from .pipeline import process_label_image
from .schemas import OCREngine

app = FastAPI(
    title="OCR Label Extraction Service",
    version="0.1.0",
    description="Extract structured data from shipping/component labels"
)

# Mount the ui folder to serve static files
ui_dir = os.path.join(os.path.dirname(__file__), "ui")
app.mount("/static", StaticFiles(directory=ui_dir), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(ui_dir, "index.html"))

@app.post("/process-label")
async def process_label(
    image_file: UploadFile = File(..., description="Upload label image"),
    engine: OCREngine = Form(..., description="OCR engine to use")
):
    try:
        image_bytes = await image_file.read()
        
        if not image_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")

        result = process_label_image(
            image_bytes=image_bytes,
            engine=engine.value
        )
        return result
    except ValueError as e:
        # e.g., "Invalid image or failed to decode", "Unsupported OCR engine"
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # e.g., Vision API Error or unhandled exceptions
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "ok"}