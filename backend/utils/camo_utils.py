# backend/utils/camo_utils.py
import numpy as np
import cv2
from PIL import Image
import base64
import io

def compute_camo_percentage(img_bgr: np.ndarray, mask: np.ndarray):
    """
    Simple composite score:
      - color similarity (histogram) between object and ring
      - texture similarity using local binary pattern approx (use Laplacian var)
      - edge difference using Canny
    Return 0..100 (higher => more camouflaged)
    """
    h,w = img_bgr.shape[:2]
    if mask.sum() == 0:
        return 0

    # object region mean color
    obj_pixels = img_bgr[mask]
    obj_mean = obj_pixels.mean(axis=0)

    # compute ring (dilate mask and subtract original)
    kernel = np.ones((31,31),np.uint8)
    ring = cv2.dilate(mask.astype(np.uint8), kernel) - mask.astype(np.uint8)
    ring_pixels = img_bgr[ring.astype(bool)]
    if ring_pixels.size == 0:
        color_dist = 1.0
    else:
        ring_mean = ring_pixels.mean(axis=0)
        color_dist = np.linalg.norm(obj_mean - ring_mean) / (np.linalg.norm(obj_mean)+1e-6)
        color_dist = np.clip(color_dist / 256.0, 0.0, 1.0)

    # texture proxy: Laplacian variance inside vs outside
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    inside_var = lap[mask].var() if mask.sum()>0 else 0.0
    outside_var = lap[~mask].var() if (~mask).sum()>0 else 0.0
    # normalize
    tex_diff = 1 - (abs(inside_var - outside_var) / (max(inside_var, outside_var, 1.0)))
    tex_diff = np.clip(tex_diff, 0.0, 1.0)

    # edge similarity: Canny density inside vs outside
    edges = cv2.Canny(gray, 100, 200)
    inside_edges = edges[mask].mean() if mask.sum()>0 else 0.0
    outside_edges = edges[~mask].mean() if (~mask).sum()>0 else 0.0
    edge_diff = 1 - (abs(inside_edges - outside_edges) / (max(inside_edges, outside_edges, 1.0)))
    edge_diff = np.clip(edge_diff, 0.0, 1.0)

    # Combine: higher similarity => higher camo score
    w_color, w_tex, w_edge = 0.45, 0.35, 0.20
    camo_score = w_color*(1-color_dist) + w_tex*tex_diff + w_edge*edge_diff
    camo_pct = int(round(100 * np.clip(camo_score, 0.0, 1.0)))
    return camo_pct

def regions_from_mask(mask, img_shape):
    # Instead of returning hundreds of regions, return only one:
    if mask.sum() == 0:
        return []

    ys, xs = np.where(mask)
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    return [{
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "intensity": 100  # mask covered = maximum camouflage
    }]

def mask_to_heatmap(mask: np.ndarray, img_bgr: np.ndarray):
    """
    Create overlay heatmap PIL image
    """
    import matplotlib.pyplot as plt
    heat = (mask.astype(np.uint8) * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_bgr, 0.6, heat_color, 0.4, 0)
    # convert BGR->RGB PIL
    out = Image.fromarray(overlay[:, :, ::-1])
    return out

def box_to_pct(box, img_shape):
    if not box:
        return None
    x1,y1,x2,y2,score,cls_id = box
    h,w,_ = img_shape
    x = int(round(100.0 * x1 / w))
    y = int(round(100.0 * y1 / h))
    width = int(round(100.0 * (x2-x1) / w))
    height = int(round(100.0 * (y2-y1) / h))
    return {"x": x, "y": y, "width": width, "height": height}
