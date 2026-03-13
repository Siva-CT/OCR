FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for OpenCV and printing support
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    cups-client \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n model
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
