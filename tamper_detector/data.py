"""Dataset discovery (CASIA v2.0 / generic folders), splitting and loading."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .ela import DEFAULT_QUALITY, DEFAULT_SCALE, compute_ela, to_rgb
from .preprocess import build_transforms, stack, to_tensor

IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
GT_SUFFIXES = ("_gt", "_mask", "-gt", "-mask")


@dataclass
class Sample:
    image: str
    mask: Optional[str]  # None => authentic (all-zero mask)
    label: int  # 0 authentic, 1 tampered


def _images_in(folder: Path) -> list[Path]:
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    )


def _strip_gt(stem: str) -> str:
    low = stem.lower()
    for suf in GT_SUFFIXES:
        if low.endswith(suf):
            return stem[: -len(suf)]
    return stem


def _find_dir(root: Path, predicate) -> Optional[Path]:
    """Shallowest directory under root (inclusive) whose lowercase name matches."""
    cands = [p for p in [root, *root.rglob("*")] if p.is_dir() and predicate(p.name.lower())]
    return min(cands, key=lambda p: len(p.parts)) if cands else None


def find_folder_dataset(
    authentic_dir: Optional[str],
    tampered_dir: str,
    mask_dir: Optional[str] = None,
) -> list[Sample]:
    """Generic layout: authentic/, tampered/ and (optionally) masks/.

    A mask is paired with a tampered image when its stem equals the image stem,
    optionally followed by _gt / _mask. Tampered images without a mask are kept
    (label 1, mask None) only when no mask_dir was given at all; otherwise they
    are dropped so pixel metrics stay honest.
    """
    samples: list[Sample] = []
    if authentic_dir:
        samples += [Sample(str(p), None, 0) for p in _images_in(Path(authentic_dir))]

    tampered = _images_in(Path(tampered_dir))
    if mask_dir is None:
        samples += [Sample(str(p), None, 1) for p in tampered]
        return samples

    masks = {_strip_gt(p.stem): p for p in _images_in(Path(mask_dir))}
    missing = 0
    for p in tampered:
        m = masks.get(p.stem)
        if m is None:
            missing += 1
            continue
        samples.append(Sample(str(p), str(m), 1))
    if missing:
        print(f"[data] {missing} tampered images had no mask in {mask_dir} and were skipped")
    return samples


def find_casia_v2(root: str) -> list[Sample]:
    """Locate Au/, Tp/ and the ground-truth folder anywhere under ``root``.

    Works with the common Kaggle re-uploads of CASIA v2.0, e.g.::

        CASIA2/Au/Au_ani_00001.jpg
        CASIA2/Tp/Tp_D_CND_M_N_ani00018_sec00096_00138.tif
        CASIA2/Groundtruth/Tp_D_CND_M_N_ani00018_sec00096_00138_gt.png
    """
    r = Path(root)
    au = _find_dir(r, lambda n: n in ("au", "authentic"))
    tp = _find_dir(r, lambda n: n in ("tp", "tampered"))
    gt = _find_dir(r, lambda n: "groundtruth" in n or "ground_truth" in n or n in ("gt", "mask", "masks"))
    if tp is None or gt is None:
        raise FileNotFoundError(
            f"Could not find CASIA 'Tp' and ground-truth folders under {root}. "
            "Use --dataset folder with explicit --authentic-dir/--tampered-dir/--mask-dir instead."
        )
    print(f"[data] CASIA layout: Au={au}  Tp={tp}  GT={gt}")
    samples = find_folder_dataset(str(au) if au else None, str(tp), str(gt))
    n_t = sum(s.label for s in samples)
    print(f"[data] {len(samples) - n_t} authentic, {n_t} tampered (with masks)")
    return samples


def split_samples(
    samples: list[Sample], val_frac: float = 0.1, test_frac: float = 0.1, seed: int = 42
) -> dict[str, list[Sample]]:
    """Stratified (by label) random split into train / val / test."""
    rng = random.Random(seed)
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for label in (0, 1):
        group = [s for s in samples if s.label == label]
        rng.shuffle(group)
        n_val = int(round(len(group) * val_frac))
        n_test = int(round(len(group) * test_frac))
        out["val"] += group[:n_val]
        out["test"] += group[n_val:n_val + n_test]
        out["train"] += group[n_val + n_test:]
    for v in out.values():
        rng.shuffle(v)
    return out


def save_split(split: dict[str, list[Sample]], path: Path) -> None:
    path.write_text(json.dumps({k: [asdict(s) for s in v] for k, v in split.items()}, indent=1))


def load_split(path: str) -> dict[str, list[Sample]]:
    raw = json.loads(Path(path).read_text())
    return {k: [Sample(**s) for s in v] for k, v in raw.items()}


def load_image(path: str) -> Image.Image:
    with Image.open(path) as im:
        return to_rgb(im).copy()


def load_mask(path: Optional[str], size: tuple[int, int]) -> np.ndarray:
    """Binary uint8 mask (1 = tampered) at image ``size`` (W, H)."""
    if path is None:
        return np.zeros((size[1], size[0]), dtype=np.uint8)
    with Image.open(path) as m:
        m = m.convert("L")
        if m.size != size:
            m = m.resize(size, Image.NEAREST)
        return (np.asarray(m) > 127).astype(np.uint8)


class TamperDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        img_size: int = 384,
        input_mode: str = "rgb+ela",
        ela_quality: int = DEFAULT_QUALITY,
        ela_scale: float = DEFAULT_SCALE,
        train: bool = False,
    ):
        self.samples = samples
        self.input_mode = input_mode
        self.ela_quality = ela_quality
        self.ela_scale = ela_scale
        self.tf = build_transforms(img_size, train)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        s = self.samples[i]
        img = load_image(s.image)
        ela = compute_ela(img, self.ela_quality, self.ela_scale)  # full resolution
        mask = load_mask(s.mask, img.size)
        out = self.tf(image=stack(np.asarray(img), ela), mask=mask)
        x = to_tensor(out["image"], self.input_mode)
        y = torch.from_numpy(out["mask"].astype(np.float32))[None]
        return x, y, torch.tensor(s.label, dtype=torch.float32)
