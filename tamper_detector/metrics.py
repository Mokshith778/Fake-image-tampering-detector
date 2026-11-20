"""Pixel-level F1 / IoU and image-level ROC-AUC / accuracy."""
from __future__ import annotations

import numpy as np
import torch


class MetricTracker:
    """Accumulates over batches.

    Pixel metrics are computed per tampered image and averaged (the standard
    protocol for CASIA): authentic images have empty masks, where F1/IoU is
    undefined. Image-level metrics use every image.
    """

    def __init__(self, pixel_threshold: float = 0.5):
        self.thr = pixel_threshold
        self.f1: list[float] = []
        self.iou: list[float] = []
        self.scores: list[float] = []
        self.labels: list[int] = []

    @torch.no_grad()
    def update(self, prob: torch.Tensor, mask: torch.Tensor, score: torch.Tensor,
               label: torch.Tensor) -> None:
        pred = (prob > self.thr).float().flatten(1)
        gt = mask.flatten(1)
        tp = (pred * gt).sum(1)
        fp = (pred * (1 - gt)).sum(1)
        fn = ((1 - pred) * gt).sum(1)
        has_gt = gt.sum(1) > 0
        f1 = 2 * tp / (2 * tp + fp + fn).clamp(min=1)
        iou = tp / (tp + fp + fn).clamp(min=1)
        self.f1 += f1[has_gt].cpu().tolist()
        self.iou += iou[has_gt].cpu().tolist()
        self.scores += score.cpu().tolist()
        self.labels += label.int().cpu().tolist()

    def compute(self, image_threshold: float | None = None) -> dict[str, float]:
        from sklearn.metrics import roc_auc_score

        s = np.asarray(self.scores)
        y = np.asarray(self.labels)
        out = {
            "pixel_f1": float(np.mean(self.f1)) if self.f1 else float("nan"),
            "pixel_iou": float(np.mean(self.iou)) if self.iou else float("nan"),
            "n_images": int(len(y)),
            "n_tampered": int(y.sum()),
        }
        if len(np.unique(y)) == 2:
            out["image_auc"] = float(roc_auc_score(y, s))
        best_t, best_acc = best_threshold(s, y)
        thr = best_t if image_threshold is None else image_threshold
        out["image_threshold"] = float(thr)
        out["image_acc"] = float(((s >= thr).astype(int) == y).mean()) if len(y) else float("nan")
        out["image_acc_best"] = float(best_acc)
        return out


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Threshold on the image score that maximises accuracy (fit on val only!)."""
    if len(scores) == 0:
        return 0.5, float("nan")
    cands = np.unique(np.concatenate([scores, [0.5]]))
    accs = [((scores >= t).astype(int) == labels).mean() for t in cands]
    i = int(np.argmax(accs))
    return float(cands[i]), float(accs[i])
