# backend/models/seg_infer.py
import numpy as np
import cv2
from ultralytics import YOLO
import os

MODEL_PATH = os.getenv("YOLO_WEIGHTS_PATH", "models/yolov8n.pt")
yolo = YOLO(MODEL_PATH)

def seg_infer(img_bgr: np.ndarray):
    """
    YOLO-guided camouflage segmentation.
    Always returns:
        - mask where the detected animal blends with surroundings
        - prob_map indicating camouflage strength
    """

    H, W = img_bgr.shape[:2]

    # Run YOLO
    results = yolo.predict(img_bgr, conf=0.25, iou=0.5, max_det=1)[0]

    if len(results.boxes) == 0:
        print("NO YOLO DETECTION")
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    # Pick strongest detection
    box = results.boxes.xyxy.cpu().numpy()[0]
    x1, y1, x2, y2 = map(int, box)

    # Extract object region
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        print("EMPTY YOLO CROP")
        return np.zeros((H, W), dtype=bool), np.zeros((H, W), dtype=np.float32)

    # Convert to LAB for texture + color uniformity
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)

    # Compute smoothness = low gradient
    grad = cv2.Laplacian(L, cv2.CV_32F)
    grad_abs = np.abs(grad)

    # Normalize
    grad_norm = cv2.normalize(grad_abs, None, 0, 1.0, cv2.NORM_MINMAX)

    # Camouflage = *low texture contrast*
    thresh = np.percentile(grad_norm, 35)
    camo_local = (grad_norm <= thresh).astype(np.float32)

    # Expand back to full mask
    mask = np.zeros((H, W), dtype=bool)
    prob = np.zeros((H, W), dtype=np.float32)

    camo_resized = cv2.resize(camo_local, (x2 - x1, y2 - y1))
    prob_resized = cv2.resize(1 - grad_norm, (x2 - x1, y2 - y1))

    mask[y1:y2, x1:x2] = camo_resized > 0.5
    prob[y1:y2, x1:x2] = prob_resized

    print("Camouflage mask pixels:", mask.sum())

    return mask, prob
