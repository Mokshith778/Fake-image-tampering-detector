"""Run the detector on your own images (e.g. a photo you edited in Photoshop/Canva).

    python predict.py --checkpoint runs/exp/best.pt my_edit.jpg other.png --out predictions/

Writes <name>_result.jpg (original | ELA | heatmap | mask) and <name>_mask.png,
and prints one JSON line per image.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from tamper_detector.data import load_image
from tamper_detector.inference import TamperDetector, side_by_side


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--pixel-threshold", type=float)
    p.add_argument("--out", default="predictions")
    a = p.parse_args()

    det = TamperDetector(a.checkpoint)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in a.images:
        img = load_image(path)
        r = det.predict(img, a.pixel_threshold)
        verdict = "TAMPERED" if r.tampered else "AUTHENTIC"
        panel = side_by_side([np.asarray(img), r.ela, r.overlay, r.mask * 255],
                             ["original", "ELA", f"{verdict} ({r.confidence:.0%})", "predicted mask"])
        stem = Path(path).stem
        cv2.imwrite(str(out / f"{stem}_result.jpg"), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(out / f"{stem}_mask.png"), r.mask * 255)
        print(json.dumps({"image": path, "verdict": verdict, "tampering_score": round(r.score, 4),
                          "confidence": round(r.confidence, 4), "threshold": round(r.threshold, 4),
                          "tampered_area_pct": round(100 * float(r.mask.mean()), 2)}))


if __name__ == "__main__":
    main()
