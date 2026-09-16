"""Utility to preprocess images for TFLite inference in AgroVeyra."""

from io import BytesIO

import numpy as np
from PIL import Image


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Preprocess raw image bytes into a normalized numpy array for TFLite inference.

    Args:
        image_bytes: Raw image file bytes (e.g., from a multipart upload).

    Returns:
        numpy.ndarray: Preprocessed image with shape (1, 224, 224, 3),
                       pixel values normalized to [0.0, 1.0].

    Raises:
        ValueError: If the image cannot be opened or decoded.
    """
    try:
        img = Image.open(BytesIO(image_bytes))
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img_array = np.asarray(img, dtype=np.float32)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"[predict.py] Image preprocessed successfully. Shape: {img_array.shape}")
        return img_array
    except Exception as e:
        print(f"[predict.py] Error preprocessing image: {e}")
        raise ValueError(f"Failed to preprocess image: {e}")


