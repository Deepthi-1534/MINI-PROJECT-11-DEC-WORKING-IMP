import cv2
import numpy as np
from ultralytics import YOLO
import torch
from torchvision.ops import nms
import os

# Load YOLO model
MODEL_PATH = os.getenv("YOLO_WEIGHTS_PATH", "models/yolov8n.pt")
print("Loading YOLO model from:", MODEL_PATH)
model = YOLO(MODEL_PATH)
print("YOLO model loaded successfully.")

def detect_infer(img):
    """
    Accepts a BGR numpy image and returns ONE bounding box:
    [x1, y1, x2, y2, score, class]
    """

    if img is None or not hasattr(img, "shape"):
        print("detect_infer received invalid image")
        return []

    H, W = img.shape[:2]

    # Run YOLO
    results = model.predict(
        img,
        conf=0.55,     # higher threshold removes leaf noise
        iou=0.45,
        max_det=10     # detect only a few objects
    )[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    # Filter out tiny & noisy detections
    min_area = (W * H) * 0.01   # MUST be >= 1% of image → removes leaf textures
    filtered = []

    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
        area = (x2 - x1) * (y2 - y1)
        if conf >= 0.55 and area >= min_area:
            filtered.append(((x1, y1, x2, y2), conf, cls))

    if not filtered:
        return []  # no detection

    # NMS
    b = torch.tensor([b[0] for b in filtered], dtype=torch.float32)
    s = torch.tensor([b[1] for b in filtered], dtype=torch.float32)
    keep = nms(b, s, iou_threshold=0.45)
    kept = [filtered[int(i)] for i in keep]

    # Select the *largest* box (most likely the hidden animal)
    best_box = None
    max_area = -1

    for (x1, y1, x2, y2), conf, cls in kept:
        area = (x2 - x1) * (y2 - y1)
        if area > max_area:
            max_area = area
            best_box = [float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)]

    return [best_box]
