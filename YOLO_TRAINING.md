# Custom YOLO Detection for SMT Reel Labels

This guide outlines the steps to build a dataset and train a dedicated YOLO model for the FASTAPI OCR extraction pipeline. 
By default, the pipeline uses a general-purpose `yolov8n.pt` object detection model, which is functional but lacks extreme precision designed expressly for physical reel label bounding.

## Structure

Your folder structure must look precisely like this:

```
dataset/
    images/
        train/  (Drop 80% of your label images here)
        val/    (Drop 20% of your label images here)
    labels/
        train/  (Text files containing bounding boxes)
        val/    (Text files containing bounding boxes)
```

## Annotation Format

Images must be annotated using the **YOLO format**. Each image (`image1.jpg`) must have a corresponding text file (`image1.txt`) inside the `labels` folder mapping bounding box predictions. 

This project expects **exactly 1 class**.

Format:
`class_id x_center y_center width height`

- `class_id`: Always `0` for our `label` class.
- Coordinates scale relatively from `0.0` to `1.0`.

Example `image1.txt`:
```text
0 0.5 0.5 0.25 0.15
0 0.5 0.8 0.25 0.15
```

## Training the Model

Once your datasets are appropriately mapped inside `dataset/`, use the provided build script to execute the training procedure:

```bash
python train_yolo.py
```

This will run for 50 epochs utilizing Ultralytics to align `yolov8n.pt` weights explicitly over your SMT images. 

Once training successfully concludes, the script saves your customized weights natively into `models/yolo-label-detector/weights/best.pt`. The backend pipeline dynamically scans and uses your custom model on next runtime if present.
