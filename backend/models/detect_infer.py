import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
import os

# Load YOLO
MODEL_PATH = os.getenv("YOLO_WEIGHTS_PATH", "backend/models/yolov8n.pt")
model = YOLO(MODEL_PATH)

# Load SINet
from models.sinet_loader import load_sinet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # backend/models/
SINET_PATH = os.path.join(BASE_DIR, "Net_epoch_best.pth")
sinet_model = load_sinet(SINET_PATH)
sinet_model.eval()

def sinet_predict(img_bgr):
    """
    Run SINet on a normal RGB/BGR image.
    MUST return mask only (HxW bool).
    """
    img = cv2.resize(img_bgr, (352, 352))
    img = img[:, :, ::-1]  # BGR → RGB
    img = img.transpose(2, 0, 1) / 255.0  # HWC → CHW
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        out = sinet_model(img)[0]  # shape 1x1xH×W → take [0]
        out = torch.sigmoid(out).squeeze().cpu().numpy()

    mask = out > 0.5
    return mask.astype(np.bool_)
    

def detect_infer(img):
    """
    Returns only YOLO bounding box.
    SINet is NOT used inside detection.
    """
    H, W = img.shape[:2]

    # YOLO detect
    results = model.predict(img, conf=0.55, iou=0.45, max_det=10)[0]

    boxes = results.boxes.xyxy.cpu().numpy()
    scores = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    filtered = []
    min_area = (W*H) * 0.01  # Remove noise

    for (x1, y1, x2, y2), conf, cls in zip(boxes, scores, classes):
        area = (x2-x1)*(y2-y1)
        if conf >= 0.55 and area >= min_area:
            filtered.append(((x1, y1, x2, y2), conf, cls))

    if not filtered:
        return []

    # NMS
    b = torch.tensor([b[0] for b in filtered], dtype=torch.float32)
    s = torch.tensor([b[1] for b in filtered], dtype=torch.float32)
    keep = nms(b, s, iou_threshold=0.45)
    kept = [filtered[int(i)] for i in keep]

    # Select the largest box
    best_box = None
    max_area = -1
    for (x1, y1, x2, y2), conf, cls in kept:
        area = (x2-x1)*(y2-y1)
        if area > max_area:
            max_area = area
            best_box = [float(x1), float(y1), float(x2), float(y2), float(conf), int(cls)]

    return [best_box]
