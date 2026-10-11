"""Identical RGB conversion, resize and normalization for training and serving."""

import numpy as np
import torch
from PIL import Image, ImageOps

from app.cnn import IMAGE_SIZE


def resize_rgb(image: Image.Image) -> np.ndarray:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.array(image, dtype=np.uint8)


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    # Input is uint8 NCHW. Fixed constants do not use validation/test statistics.
    return images.float().div(255.0).sub(0.5).div(0.5)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    pixels = torch.from_numpy(resize_rgb(image)).permute(2, 0, 1)
    return normalize_batch(pixels.unsqueeze(0))
