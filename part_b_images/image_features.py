from pathlib import Path
from typing import Union

import cv2
import numpy as np

DEFAULT_IMG_SIZE = (128, 128)
DEFAULT_BINS = 32
FEATURE_DIM = DEFAULT_BINS * 3


def _prepare_three_channel_image(
    image_input: Union[str, Path, np.ndarray],
    is_rgb: bool,
) -> tuple[np.ndarray, int]:
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        return img, cv2.COLOR_BGR2HSV

    if not isinstance(image_input, np.ndarray):
        raise TypeError(
            "image_input must be a file path (str/Path) or a numpy.ndarray."
        )

    if image_input.size == 0:
        raise ValueError("Input image array is empty.")

    img = image_input
    if np.issubdtype(img.dtype, np.floating):
        max_val = float(np.max(img))
        if max_val <= 1.0:
            img = np.clip(img * 255.0, 0.0, 255.0).round().astype(np.uint8)
        else:
            img = np.clip(img, 0.0, 255.0).round().astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img, cv2.COLOR_BGR2HSV

    if img.ndim == 3 and img.shape[2] == 1:
        img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        return img, cv2.COLOR_BGR2HSV

    if img.ndim == 3 and img.shape[2] == 4:
        drop_alpha_code = cv2.COLOR_RGBA2RGB if is_rgb else cv2.COLOR_BGRA2BGR
        img = cv2.cvtColor(img, drop_alpha_code)
    elif img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"Unsupported image array shape: {img.shape}")

    cvt_code = cv2.COLOR_RGB2HSV if is_rgb else cv2.COLOR_BGR2HSV
    return img, cvt_code


def extract_flower_features(
    image_input: Union[str, Path, np.ndarray],
    img_size: tuple[int, int] = DEFAULT_IMG_SIZE,
    bins: int = DEFAULT_BINS,
    is_rgb: bool = False,
) -> np.ndarray:
    img, cvt_code = _prepare_three_channel_image(image_input, is_rgb=is_rgb)

    resized = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(resized, cvt_code)
    hsv_norm = hsv.astype(np.float32) / 255.0

    channel_hists = []
    for channel_idx in range(3):
        hist, _ = np.histogram(
            hsv_norm[:, :, channel_idx],
            bins=bins,
            range=(0.0, 1.0),
        )
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-7)
        channel_hists.append(hist)

    return np.concatenate(channel_hists).astype(np.float32)
