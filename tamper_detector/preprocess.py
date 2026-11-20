"""Turn an (RGB, ELA) pair into a network input tensor.

Used by BOTH training and inference so the two can never drift apart.
"""
from __future__ import annotations

import albumentations as A
import numpy as np
import torch

# RGB channels: ImageNet statistics (the encoder was pre-trained on them).
# ELA channels: mostly near zero, so a simple 0.5 / 0.5 centring is enough.
MEAN = np.array([0.485, 0.456, 0.406, 0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225, 0.5, 0.5, 0.5], dtype=np.float32)

# Which of the 6 stacked channels each input mode keeps.
INPUT_MODES = {
    "rgb+ela": [0, 1, 2, 3, 4, 5],
    "ela": [3, 4, 5],
    "rgb": [0, 1, 2],
}


def in_channels(input_mode: str) -> int:
    return len(INPUT_MODES[input_mode])


def build_transforms(img_size: int, train: bool) -> A.Compose:
    """Geometric transforms applied jointly to the 6-channel stack and mask.

    No colour / blur / JPEG augmentation: those would alter the RGB channels
    after ELA was computed and teach the model wrong correlations.
    """
    ops: list = [A.Resize(img_size, img_size)]
    if train:
        ops += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.3),
        ]
    return A.Compose(ops)


def stack(rgb: np.ndarray, ela: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 + HxWx3 uint8 -> HxWx6 uint8."""
    return np.concatenate([rgb, ela], axis=2)


def to_tensor(x6: np.ndarray, input_mode: str) -> torch.Tensor:
    """HxWx6 uint8 -> CxHxW float tensor, normalised, channels selected."""
    x = x6.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    x = x[:, :, INPUT_MODES[input_mode]]
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
