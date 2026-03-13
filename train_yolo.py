import os
from ultralytics import YOLO

def train():
    # Validate dataset configuration exists
    dataset_yaml = os.path.abspath("dataset/labels.yaml")
    if not os.path.exists(dataset_yaml):
        raise FileNotFoundError(f"Dataset config not found: {dataset_yaml}")
    
    # Load the base model (downloads automatically if missing)
    print("Loading base yolov8n.pt model...")
    model = YOLO("yolov8n.pt")

    print(f"Starting training on dataset configuration: {dataset_yaml}")
    
    # Train the model targeting our custom single-class SMT label format
    results = model.train(
        data=dataset_yaml,
        epochs=int(os.environ.get("YOLO_EPOCHS", 50)),
        imgsz=int(os.environ.get("YOLO_IMGSZ", 640)),
        batch=int(os.environ.get("YOLO_BATCH", 8)),
        project="models",
        name="yolo-label-detector",
        exist_ok=True  # Overwrites existing folder with the same name during retraining
    )
    
    print("\nTraining completed. Custom model saved to models/yolo-label-detector/weights/best.pt")

if __name__ == "__main__":
    train()
