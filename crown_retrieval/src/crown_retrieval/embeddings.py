"""DINOv2 embedding extraction from polygon-masked tree crown crops."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from rasterio.mask import mask
from shapely.geometry import mapping
from torchvision import transforms

from . import IMAGENET_MEAN, IMAGENET_STD


def resolve_device(requested_device: str) -> torch.device:
    """Use the requested CUDA device when available, otherwise fall back to CPU."""
    requested = torch.device(requested_device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("CUDA is unavailable; using CPU.")
        return torch.device("cpu")
    return requested


def load_dinov2(model_name: str, device: torch.device) -> torch.nn.Module:
    """Load a pretrained DINOv2 backbone and switch it to inference mode."""
    model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=True, verbose=False)
    return model.to(device).eval()


def preprocessing(image_size: int) -> transforms.Compose:
    """Match the ImageNet normalization used by DINOv2 feature extraction."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def polygon_crop(geometry, raster, background_value: int) -> Image.Image | None:
    """Crop a polygon and replace non-crown pixels with a constant neutral gray."""
    try:
        masked, _ = mask(raster, [mapping(geometry)], crop=True, filled=False)
    except Exception as error:
        print(f"[WARNING] Could not crop a candidate: {error}")
        return None
    if masked.shape[0] < 3:
        return None
    rgb = masked[:3]
    data = np.asarray(rgb.data)
    alpha_mask = np.ma.getmaskarray(rgb)
    background = np.full_like(data, background_value)
    image = np.where(alpha_mask, background, data).transpose(1, 2, 0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(image, mode="RGB")


@torch.inference_mode()
def embed_crop(image: Image.Image, model: torch.nn.Module, transform, device: torch.device) -> np.ndarray:
    """Return a single L2-normalized DINOv2 embedding as float32."""
    tensor = transform(image).unsqueeze(0).to(device)
    features = torch.nn.functional.normalize(model(tensor), dim=1)
    return features.squeeze(0).cpu().numpy().astype(np.float32)

