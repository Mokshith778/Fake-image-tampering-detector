"""Evaluate a checkpoint: pixel F1 / IoU, image ROC-AUC / accuracy, example panels.

In-distribution (held-out CASIA test split written by train.py):
    python evaluate.py --checkpoint runs/exp/best.pt --split runs/exp/split.json

Cross-dataset (e.g. Columbia Uncompressed: 4cam_auth/ vs 4cam_splc/):
    python evaluate.py --checkpoint runs/exp/best.pt \
        --authentic-dir columbia/4cam_auth --tampered-dir columbia/4cam_splc
    (add --mask-dir with binary masks for pixel metrics; without it only
     image-level metrics are reported)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from tamper_detector.data import (TamperDataset, find_casia_v2, find_folder_dataset, load_image,
                                  load_mask, load_split)
from tamper_detector.inference import TamperDetector, pick_device, side_by_side
from tamper_detector.metrics import MetricTracker
from tamper_detector.model import image_score, load_checkpoint


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", help="split.json from train.py")
    p.add_argument("--subset", default="test", choices=["train", "val", "test"])
    p.add_argument("--casia-root")
    p.add_argument("--authentic-dir")
    p.add_argument("--tampered-dir")
    p.add_argument("--mask-dir")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--examples", type=int, default=8, help="number of visual panels to save")
    p.add_argument("--out", help="output dir (default: next to the checkpoint)")
    a = p.parse_args()

    if a.split:
        samples, name = load_split(a.split)[a.subset], f"{Path(a.split).parent.name}_{a.subset}"
    elif a.casia_root:
        samples, name = find_casia_v2(a.casia_root), "casia_all"
    elif a.tampered_dir:
        samples, name = find_folder_dataset(a.authentic_dir, a.tampered_dir, a.mask_dir), Path(a.tampered_dir).parent.name
    else:
        raise SystemExit("give --split, --casia-root or --tampered-dir")

    device = pick_device()
    model, ckpt = load_checkpoint(a.checkpoint, device)
    cfg = ckpt["config"]
    ds = TamperDataset(samples, cfg["img_size"], cfg["input_mode"], cfg["ela_quality"], cfg["ela_scale"])
    dl = DataLoader(ds, batch_size=a.batch_size, num_workers=a.workers)

    tracker = MetricTracker(float(ckpt.get("pixel_threshold", 0.5)))
    with torch.no_grad():
        for x, y, label in dl:
            prob = torch.sigmoid(model(x.to(device)).float())
            tracker.update(prob, y.to(device), image_score(prob), label)

    has_masks = any(s.mask for s in samples)
    m = tracker.compute(image_threshold=float(ckpt.get("image_threshold", 0.5)))
    if not has_masks:
        m["pixel_f1"] = m["pixel_iou"] = None  # no ground truth masks -> undefined
    out = Path(a.out) if a.out else Path(a.checkpoint).parent / f"eval_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))

    # Visual examples: a mix of tampered and authentic images.
    det = TamperDetector(a.checkpoint, device)
    tam = [s for s in samples if s.label == 1][: (a.examples + 1) // 2]
    au = [s for s in samples if s.label == 0][: a.examples // 2]
    for i, s in enumerate(tam + au):
        img = load_image(s.image)
        pred = det.predict(img)
        gt = load_mask(s.mask, img.size) * 255
        verdict = f"{'TAMPERED' if pred.tampered else 'AUTHENTIC'} {pred.score:.2f}"
        panel = side_by_side([np.asarray(img), pred.ela, gt, pred.overlay],
                             ["original" + (" (tampered)" if s.label else " (authentic)"),
                              "ELA", "ground truth", verdict])
        cv2.imwrite(str(out / f"example_{i:02d}.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"[eval] metrics + {len(tam + au)} example panels written to {out}")


if __name__ == "__main__":
    main()
