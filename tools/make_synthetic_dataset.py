"""Build a tiny CASIA-style dataset (Au/, Tp/, Groundtruth/) with synthetic forgeries.

Useful to check the whole pipeline end to end in minutes, before downloading
CASIA, and as a sanity check that the model can learn ELA cues at all:

    python tools/make_synthetic_dataset.py --out data/synthetic --n 200
    python train.py --data-root data/synthetic --epochs 3 --img-size 256 --out runs/synthetic

Forgeries: splice (patch from another image, different JPEG history) and
copy-move (patch from the same image, re-compressed), each saved as JPEG.
"""
from __future__ import annotations

import argparse
import io
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def random_scene(rng: random.Random, w: int, h: int) -> Image.Image:
    """Smooth gradients + shapes + sensor-like noise: enough texture for ELA."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.stack([
        rng.uniform(40, 200) + rng.uniform(-60, 60) * xx / w + rng.uniform(-60, 60) * yy / h
        for _ in range(3)], axis=2)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(4, 10)):
        x0, y0 = rng.randrange(w), rng.randrange(h)
        x1, y1 = x0 + rng.randint(20, w // 2), y0 + rng.randint(20, h // 2)
        color = tuple(rng.randrange(256) for _ in range(3))
        (d.ellipse if rng.random() < 0.5 else d.rectangle)([x0, y0, x1, y1], fill=color)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2)))
    arr = np.asarray(img).astype(np.float32) + np.random.default_rng(rng.randrange(1 << 30)).normal(0, 4, (h, w, 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def jpeg(img: Image.Image, q: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/synthetic")
    p.add_argument("--n", type=int, default=200, help="images per class")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.out)
    for sub in ("Au", "Tp", "Groundtruth"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    for i in range(a.n):
        w, h = rng.choice([(384, 256), (256, 384), (320, 320)])
        # Authentic: one capture, one compression.
        jpeg(random_scene(rng, w, h), rng.randint(70, 95)).save(root / "Au" / f"Au_syn_{i:05d}.jpg", quality=95)

        # Tampered.
        host = jpeg(random_scene(rng, w, h), rng.randint(60, 80))
        pw, ph = rng.randint(w // 6, w // 2), rng.randint(h // 6, h // 2)
        if rng.random() < 0.5:  # splice from a different image, different compression
            donor = jpeg(random_scene(rng, w, h), rng.randint(90, 100))
            kind = "S"
        else:  # copy-move inside the same image, re-compressed
            donor = jpeg(host, rng.randint(90, 100))
            kind = "C"
        sx, sy = rng.randrange(w - pw), rng.randrange(h - ph)
        dx, dy = rng.randrange(w - pw), rng.randrange(h - ph)
        patch = donor.crop((sx, sy, sx + pw, sy + ph))
        shape = Image.new("L", (pw, ph), 0)
        ImageDraw.Draw(shape).ellipse([0, 0, pw - 1, ph - 1], fill=255) if rng.random() < 0.5 \
            else ImageDraw.Draw(shape).rectangle([0, 0, pw - 1, ph - 1], fill=255)
        host.paste(patch, (dx, dy), shape)
        mask = Image.new("L", (w, h), 0)
        mask.paste(shape, (dx, dy))
        name = f"Tp_{kind}_syn_{i:05d}"
        host.save(root / "Tp" / f"{name}.jpg", quality=95)
        mask.save(root / "Groundtruth" / f"{name}_gt.png")
    print(f"wrote {a.n} authentic + {a.n} tampered images to {root}")


if __name__ == "__main__":
    main()
