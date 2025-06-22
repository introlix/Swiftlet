# This file is was taken from https://github.com/google/gemma_pytorch/blob/main/gemma/siglip_vision/preprocessor.py
# This is under the Apache License 2.0

from collections.abc import Sequence

from PIL import Image
import torch
import numpy as np  # Import NumPy

_IMAGE_MEAN = [0.5, 0.5, 0.5]  # equivalent to 127.5/255
_IMAGE_STD = [0.5, 0.5, 0.5]  # equivalent to 127.5/255
DEFAULT_IMAGE_SIZE = 896


def preprocess_images_for_siglip_vision(
    images: Sequence[Image.Image], image_size=DEFAULT_IMAGE_SIZE
) -> torch.Tensor:
    """Preprocesses a list of PIL images for Siglip vision model using only PyTorch and PIL.

    Args:
        images: A sequence of PIL Image objects.
        image_size: The target size for resizing the images.

    Returns:
        A sequence of torch.Tensor objects, each of shape (C, H, W).
    """
    processed_images = []

    mean_tensor = torch.tensor(_IMAGE_MEAN, dtype=torch.float32).reshape(3, 1, 1)
    std_tensor = torch.tensor(_IMAGE_STD, dtype=torch.float32).reshape(3, 1, 1)

    for image in images:
        # Resize image
        image = image.resize((image_size, image_size), Image.Resampling.BILINEAR)

        # Convert to NumPy and ensure float32 type
        image_np = np.array(image, dtype=np.float32) / 255.0  # Normalize to [0,1]

        # Convert to PyTorch tensor and rearrange channels
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1)  # (H, W, C) → (C, H, W)

        # Normalize
        image_tensor = (image_tensor - mean_tensor) / std_tensor

        # Clip the values to [-1, 1]
        image_tensor = torch.clamp(image_tensor, -1, 1)

        processed_images.append(image_tensor)

    return processed_images