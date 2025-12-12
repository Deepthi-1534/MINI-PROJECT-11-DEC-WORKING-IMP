# backend/app.py
from dotenv import load_dotenv
load_dotenv()

import os
import io
import base64
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import requests
from PIL import Image
import numpy as np

# model wrappers
from models.seg_infer import seg_infer
from models.detect_infer import detect_infer
from models.classify_infer import classify_infer
from models.llm_infer import generate_description

from utils.io_utils import read_imagefile, pil_to_bytes
from utils.camo_utils import compute_camo_percentage, regions_from_mask, mask_to_heatmap, box_to_pct

app = FastAPI()

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeResponse(BaseModel):
    detected: bool
    species: Optional[str]
    camouflagePercentage: int
    confidence: int
    description: str
    adaptations: list
    boundingBox: Optional[dict]
    camouflageRegions: list


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(image_url: Optional[str] = Form(None), file: Optional[UploadFile] = File(None)):
    """
    Accept either multipart file upload (file) or an image_url form field.
    Returns JSON with segmentation mask-derived statistics, bounding box, species guess, etc.
    """

    if (not image_url) and (not file):
        raise HTTPException(status_code=400, detail="Provide file or image_url")

    # Read image bytes
    try:
        if image_url:
            r = requests.get(image_url, timeout=10)
            r.raise_for_status()
            contents = r.content
        else:
            contents = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch/read image: {e}")

    # quick check
    if not contents or len(contents) == 0:
        raise HTTPException(status_code=400, detail="Empty image uploaded")

    # Decode with PIL and validate
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Convert PIL -> numpy (RGB)
    img_np = np.array(img)
    if img_np is None or img_np.size == 0:
        raise HTTPException(status_code=400, detail="Failed to decode image to numpy array")

    # Convert RGB -> BGR for OpenCV / YOLO (your models expect BGR)
    img_np = img_np[:, :, ::-1].copy()

    # ----------------------------------------------------------------
    # 1) Detector (detect_infer now returns dict: {'box':..., 'sinet_mask':...})
    # ----------------------------------------------------------------
    try:
        det_result = detect_infer(img_np)
        if det_result is None:
            det_result = {"box": None, "sinet_mask": None}
        # Normalize to old 'boxes' list format for downstream compatibility
        boxes = []
        if isinstance(det_result, dict):
            box = det_result.get("box", None)
            sinet_mask_from_detector = det_result.get("sinet_mask", None)
            if box is not None:
                # old expectation: list of boxes [[x1,y1,x2,y2,score,cls], ...]
                boxes = [box]
        else:
            # backward compatibility: detect_infer returned list of boxes previously
            boxes = det_result if isinstance(det_result, list) else []
            sinet_mask_from_detector = None
        if boxes is None:
            boxes = []
    except Exception as e:
        print("detect_infer error:", e)
        raise HTTPException(status_code=500, detail=f"Detection error: {e}")

    # ----------------------------------------------------------------
    # 2) Segmentation
    #    Priority order:
    #      1) If detect_infer returned a sinet_mask, use that.
    #      2) Else call seg_infer() (existing segmentation module).
    #    seg_infer may return (mask, prob_map) or a single mask.
    # ----------------------------------------------------------------
    try:
        mask = None
        prob_map = None

        if sinet_mask_from_detector is not None:
            # detector returned a SINet mask - normalize it
            sinet_mask = sinet_mask_from_detector
            # convert to numpy if torch tensor or PIL
            if hasattr(sinet_mask, "cpu"):
                sinet_mask = sinet_mask.cpu().numpy()
            if isinstance(sinet_mask, np.ndarray):
                # If mask is 0-255, scale to 0-1 float for prob_map and boolean for mask
                if sinet_mask.dtype == np.uint8 or sinet_mask.max() > 1:
                    prob_map = (sinet_mask.astype(float) / 255.0).astype(float)
                    mask = (prob_map > 0.5)
                else:
                    prob_map = sinet_mask.astype(float)
                    mask = sinet_mask.astype(bool)
            else:
                # try to coerce via PIL
                try:
                    from PIL import Image as PILImage
                    arr = np.array(PILImage.fromarray(sinet_mask))
                    prob_map = (arr.astype(float) / 255.0).astype(float) if arr.max() > 1 else arr.astype(float)
                    mask = prob_map > 0.5
                except Exception:
                    mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
                    prob_map = np.zeros_like(mask, dtype=float)
        else:
            # fallback to existing seg_infer
            seg_out = seg_infer(img_np)
            if isinstance(seg_out, tuple) and len(seg_out) == 2:
                mask, prob_map = seg_out
            else:
                mask = seg_out
                prob_map = None
                if mask is not None:
                    # If it's numeric mask, convert to float prob_map
                    if mask.dtype == np.uint8 or mask.max() > 1:
                        prob_map = (mask.astype(float) / 255.0).astype(float)
                        mask = (prob_map > 0.5)
                    else:
                        prob_map = mask.astype(float)
                        mask = mask.astype(bool)

        # Ensure shapes and types
        if mask is None:
            mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
        if prob_map is None:
            prob_map = np.zeros_like(mask, dtype=float)

        if mask.dtype != bool:
            mask_bool = mask.astype(bool)
        else:
            mask_bool = mask

    except Exception as e:
        print("seg_infer / mask normalization error:", e)
        mask = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=bool)
        prob_map = np.zeros((img_np.shape[0], img_np.shape[1]), dtype=float)
        mask_bool = mask

    # ----------------------------------------------------------------
    # 3) Choose primary bounding box (if any) - prefer detection that overlaps mask
    #    The rest of this logic is unchanged from your original file.
    # ----------------------------------------------------------------
    primary_box = None
    if boxes:
        best = None
        best_score = -1
        for b in boxes:
            try:
                x1, y1, x2, y2, score, cls_id = b
                bx1, by1, bx2, by2 = map(int, [x1, y1, x2, y2])
                # clamp coords
                bx1 = max(0, min(bx1, img_np.shape[1] - 1))
                bx2 = max(0, min(bx2, img_np.shape[1] - 1))
                by1 = max(0, min(by1, img_np.shape[0] - 1))
                by2 = max(0, min(by2, img_np.shape[0] - 1))
                if bx2 <= bx1 or by2 <= by1:
                    overlap = 0
                else:
                    mask_crop = mask_bool[by1:by2, bx1:bx2]
                    overlap = int(mask_crop.sum()) if mask_crop.size > 0 else 0
            except Exception:
                overlap = 0
            if overlap > best_score:
                best_score = overlap
                best = b
        primary_box = best if best_score > 0 else boxes[0]

    # If no boxes, compute bbox from mask
    if (not primary_box) and mask_bool.sum() > 0:
        ys, xs = np.where(mask_bool)
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        primary_box = [x1, y1, x2, y2, 0.95, 0]  # synthetic score

    # ----------------------------------------------------------------
    # 4) Species classification (on crop if primary_box exists)
    # ----------------------------------------------------------------
    species = None
    species_prob = 0.0
    if primary_box:
        try:
            x1, y1, x2, y2, _, _ = map(int, primary_box)
            # clamp coords
            x1 = max(0, min(x1, img_np.shape[1] - 1))
            x2 = max(0, min(x2, img_np.shape[1] - 1))
            y1 = max(0, min(y1, img_np.shape[0] - 1))
            y2 = max(0, min(y2, img_np.shape[0] - 1))
            crop = img_np[y1:y2, x1:x2] if (y2 > y1 and x2 > x1) else img_np
            # classification might expect RGB; convert if needed by your classify_infer
            species, species_prob = classify_infer(crop)
            if species is None:
                species = None
                species_prob = 0.0
        except Exception as e:
            print("classify_infer error:", e)
            species = None
            species_prob = 0.0

    # ----------------------------------------------------------------
    # 5) camo percentage (algorithmic)
    # ----------------------------------------------------------------
    try:
        camo_pct = int(round(compute_camo_percentage(img_np, mask_bool)))
    except Exception as e:
        print("compute_camo_percentage error:", e)
        camo_pct = 0

    # ----------------------------------------------------------------
    # 6) confidence (combine mask mean prob + species prob + detection confidence)
    # ----------------------------------------------------------------
    try:
        mask_conf = float(prob_map[mask_bool].mean()) if mask_bool.sum() > 0 else 0.0
    except Exception:
        mask_conf = 0.0
    detect_conf = float(primary_box[4]) if primary_box else 0.0
    combined_conf = (0.5 * mask_conf + 0.3 * species_prob + 0.2 * detect_conf)
    confidence = int(round(max(0, min(1, combined_conf)) * 100))

    # ----------------------------------------------------------------
    # 7) regions
    # ----------------------------------------------------------------
    try:
        regions = regions_from_mask(mask_bool, img_np.shape)
    except Exception as e:
        print("regions_from_mask error:", e)
        regions = []

    # ----------------------------------------------------------------
    # 8) description via LLM (optional, may fail)
    # ----------------------------------------------------------------
    try:
        desc, adaptations = generate_description({
            "species": species,
            "species_prob": species_prob,
            "camo_pct": camo_pct,
            "regions": regions,
            "bbox": box_to_pct(primary_box, img_np.shape) if primary_box else None
        })
    except Exception as e:
        print("generate_description error:", e)
        desc = ""
        adaptations = []

    response = {
        "detected": (camo_pct > 0) or (primary_box is not None),
        "species": species,
        "camouflagePercentage": camo_pct,
        "confidence": confidence,
        "description": desc,
        "adaptations": adaptations,
        "boundingBox": box_to_pct(primary_box, img_np.shape) if primary_box else None,
        "camouflageRegions": regions
    }
    return response


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("app:app", host="0.0.0.0", port=port, log_level="info")
