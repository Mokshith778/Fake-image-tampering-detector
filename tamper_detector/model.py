"""Segmentation model, loss, image-level score and checkpoint I/O."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

ARCHS = {
    "unet": "Unet",
    "unetplusplus": "UnetPlusPlus",
    "fpn": "FPN",
    "deeplabv3plus": "DeepLabV3Plus",
}


def build_model(arch: str = "unet", encoder: str = "resnet34", in_channels: int = 6,
                pretrained: bool = True) -> nn.Module:
    """U-Net (or other SMP decoder) with an ImageNet encoder.

    With in_channels=6, SMP adapts the pretrained first conv: the extra ELA
    channels get initialised from the RGB filters, so pre-training still helps.
    """
    import segmentation_models_pytorch as smp

    cls = getattr(smp, ARCHS[arch])
    return cls(
        encoder_name=encoder,
        encoder_weights="imagenet" if pretrained else None,
        in_channels=in_channels,
        classes=1,
    )


class BCEDiceLoss(nn.Module):
    """BCE (pixel-wise calibration) + Dice (robust to tiny tampered regions)."""

    def __init__(self, bce_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.w = bce_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = self.bce(logits, target)
        p = torch.sigmoid(logits).flatten(1)
        t = target.flatten(1)
        # Dice over the whole batch, so authentic (all-zero) masks don't produce
        # undefined per-image Dice and the loss stays well behaved.
        inter = (p * t).sum()
        dice = 1 - (2 * inter + self.smooth) / (p.sum() + t.sum() + self.smooth)
        return self.w * bce + (1 - self.w) * dice


def image_score(prob: torch.Tensor, top_frac: float = 0.005) -> torch.Tensor:
    """Image-level tampering score from a probability map (B,1,H,W) -> (B,).

    "Peak activation", made robust: the mean of the top 0.5% most suspicious
    pixels instead of the single max pixel, which is noisy.
    """
    flat = prob.flatten(1)
    k = max(1, int(flat.shape[1] * top_frac))
    return flat.topk(k, dim=1).values.mean(dim=1)


def save_checkpoint(path: Path, model: nn.Module, config: dict[str, Any], **extra: Any) -> None:
    torch.save({"state_dict": model.state_dict(), "config": config, **extra}, path)


def load_checkpoint(path: str, device: str | torch.device = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    from .preprocess import in_channels

    model = build_model(cfg["arch"], cfg["encoder"], in_channels(cfg["input_mode"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt
