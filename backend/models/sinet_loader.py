import os
import torch
from .sinet_model import SINetV2

def load_sinet(weights_path):
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Cannot find SINet weights at: {weights_path}")

    state = torch.load(weights_path, map_location="cpu")
    model = SINetV2()
    model.load_state_dict(state, strict=False)
    model.eval()
    return model
