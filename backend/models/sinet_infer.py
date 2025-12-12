import torch
import cv2
import numpy as np
import torch.nn.functional as F


def sinet_infer(model, img_bgr):
    """
    img_bgr: numpy array in BGR
    returns: mask (H,W) uint8 [0..255]
    """

    H, W = img_bgr.shape[:2]

    # Convert BGR → RGB → Tensor
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        pred = model(tensor)
        pred = torch.sigmoid(pred)

    pred = pred.squeeze().cpu().numpy()
    pred = (pred * 255).astype('uint8')

    return cv2.resize(pred, (W, H))
