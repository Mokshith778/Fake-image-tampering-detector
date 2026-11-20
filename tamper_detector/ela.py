"""Error Level Analysis (ELA).

Re-save an image as JPEG at a known quality and measure how much every pixel
changes. Regions that share the image's compression history change by a
similar, small amount; pasted, cloned or regenerated regions usually have a
different history and stand out.

ELA must be computed on the image at its ORIGINAL resolution, before any
resizing: resampling destroys the 8x8 JPEG block artefacts ELA relies on.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

DEFAULT_QUALITY = 90
DEFAULT_SCALE = 15.0


def to_rgb(img: Image.Image) -> Image.Image:
    """Convert any PIL mode (RGBA, P, L, CMYK, I;16 ...) to 8-bit RGB.

    Transparent pixels are composited onto white so they don't become black
    blocks that look like tampering.
    """
    if img.mode == "RGB":
        return img
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode.startswith("I;16") or img.mode == "I":
        arr = np.asarray(img, dtype=np.float32)
        arr = 255.0 * (arr - arr.min()) / max(float(arr.max() - arr.min()), 1.0)
        return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def compute_ela(
    img: Image.Image,
    quality: int = DEFAULT_QUALITY,
    scale: float = DEFAULT_SCALE,
) -> np.ndarray:
    """Return the ELA map of ``img`` as an HxWx3 uint8 array.

    ``scale`` is a FIXED amplification (not per-image max-stretching) so the
    network sees error levels on a consistent scale across images.
    """
    rgb = to_rgb(img)
    buf = io.BytesIO()
    rgb.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")

    a = np.asarray(rgb, dtype=np.int16)
    b = np.asarray(resaved, dtype=np.int16)
    diff = np.abs(a - b).astype(np.float32) * scale
    return np.clip(diff, 0, 255).astype(np.uint8)


def ela_for_display(ela: np.ndarray) -> np.ndarray:
    """Contrast-stretch an ELA map for human viewing (not for the model)."""
    hi = float(np.percentile(ela, 99.5))
    if hi <= 0:
        return ela
    out = ela.astype(np.float32) * (255.0 / hi)
    return np.clip(out, 0, 255).astype(np.uint8)
