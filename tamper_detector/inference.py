"""Single-image inference + visualisation, shared by predict.py and app.py."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image

from .ela import compute_ela, ela_for_display, to_rgb
from .model import image_score, load_checkpoint
from .preprocess import build_transforms, stack, to_tensor


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Prediction:
    tampered: bool
    score: float           # image-level tampering score in [0, 1]
    threshold: float       # image-level decision threshold (calibrated on val)
    prob_map: np.ndarray   # HxW float32, original resolution
    mask: np.ndarray       # HxW uint8 {0,1}
    ela: np.ndarray        # HxWx3 uint8, contrast-stretched for display
    overlay: np.ndarray    # HxWx3 uint8 heatmap blended on the image

    @property
    def confidence(self) -> float:
        """Confidence in the verdict given, in [0.5, 1].

        Measured as distance from the calibrated decision threshold (which is
        usually not 0.5), so a score just past the threshold reads ~50%.
        """
        t = min(max(self.threshold, 1e-6), 1 - 1e-6)
        if self.tampered:
            return 0.5 + 0.5 * (self.score - t) / (1 - t)
        return 0.5 + 0.5 * (t - self.score) / t


class TamperDetector:
    def __init__(self, checkpoint: str, device: str | None = None):
        self.device = device or pick_device()
        self.model, ckpt = load_checkpoint(checkpoint, self.device)
        self.cfg = ckpt["config"]
        self.image_threshold = float(ckpt.get("image_threshold", 0.5))
        self.pixel_threshold = float(ckpt.get("pixel_threshold", 0.5))
        self.tf = build_transforms(self.cfg["img_size"], train=False)

    @torch.no_grad()
    def predict(self, img: Image.Image, pixel_threshold: float | None = None) -> Prediction:
        img = to_rgb(img)
        rgb = np.asarray(img)
        ela = compute_ela(img, self.cfg["ela_quality"], self.cfg["ela_scale"])
        x = to_tensor(self.tf(image=stack(rgb, ela))["image"], self.cfg["input_mode"])
        prob = torch.sigmoid(self.model(x[None].to(self.device)))
        score = float(image_score(prob)[0])

        h, w = rgb.shape[:2]
        prob_map = cv2.resize(prob[0, 0].float().cpu().numpy(), (w, h), interpolation=cv2.INTER_LINEAR)
        thr = self.pixel_threshold if pixel_threshold is None else pixel_threshold
        mask = (prob_map > thr).astype(np.uint8)
        return Prediction(
            tampered=score >= self.image_threshold,
            score=score,
            threshold=self.image_threshold,
            prob_map=prob_map,
            mask=mask,
            ela=ela_for_display(ela),
            overlay=make_overlay(rgb, prob_map, mask),
        )


def make_overlay(rgb: np.ndarray, prob_map: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    """JET heatmap blended where the model is suspicious, plus red mask outlines."""
    heat = cv2.applyColorMap((np.clip(prob_map, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    a = (alpha * np.clip(prob_map * 1.5, 0, 1))[..., None]  # fade out low-probability areas
    out = (rgb * (1 - a) + heat * a).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    thick = max(2, round(max(rgb.shape[:2]) / 300))
    cv2.drawContours(out, contours, -1, (255, 0, 0), thick)
    return out


def side_by_side(images: list[np.ndarray], titles: list[str], height: int = 384) -> np.ndarray:
    """Horizontal panel with a title bar on each tile (for reports / CLI output)."""
    tiles = []
    for im, title in zip(images, titles):
        if im.ndim == 2:
            im = np.repeat((im * 255 if im.max() <= 1 else im).astype(np.uint8)[..., None], 3, axis=2)
        w = round(im.shape[1] * height / im.shape[0])
        tile = cv2.resize(im, (w, height), interpolation=cv2.INTER_AREA)
        bar = np.full((28, w, 3), 255, np.uint8)
        cv2.putText(bar, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, tile]))
        tiles.append(np.full((height + 28, 6, 3), 255, np.uint8))
    return np.hstack(tiles[:-1])
