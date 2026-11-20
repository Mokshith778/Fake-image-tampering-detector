"""Train the ELA + U-Net tampering segmenter.

Example (Colab T4, CASIA v2.0):
    python train.py --dataset casia --data-root /content/casia --epochs 20 --out runs/casia_r34
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from tamper_detector.data import (TamperDataset, find_casia_v2, find_folder_dataset,
                                  save_split, split_samples)
from tamper_detector.inference import pick_device
from tamper_detector.metrics import MetricTracker
from tamper_detector.model import BCEDiceLoss, build_model, image_score, save_checkpoint
from tamper_detector.preprocess import INPUT_MODES, in_channels


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = p.add_argument_group("data")
    d.add_argument("--dataset", choices=["casia", "folder"], default="casia")
    d.add_argument("--data-root", help="CASIA v2.0 root (any folder containing Au/, Tp/, Groundtruth/)")
    d.add_argument("--authentic-dir")
    d.add_argument("--tampered-dir")
    d.add_argument("--mask-dir")
    d.add_argument("--val-frac", type=float, default=0.1)
    d.add_argument("--test-frac", type=float, default=0.1)
    d.add_argument("--limit", type=int, default=0, help="use only N samples (quick debugging)")
    m = p.add_argument_group("model")
    m.add_argument("--arch", default="unet", choices=["unet", "unetplusplus", "fpn", "deeplabv3plus"])
    m.add_argument("--encoder", default="resnet34", help="any SMP encoder, e.g. efficientnet-b0")
    m.add_argument("--input-mode", default="rgb+ela", choices=list(INPUT_MODES))
    m.add_argument("--no-pretrained", action="store_true")
    m.add_argument("--img-size", type=int, default=384)
    m.add_argument("--ela-quality", type=int, default=90)
    m.add_argument("--ela-scale", type=float, default=15.0)
    t = p.add_argument_group("training")
    t.add_argument("--epochs", type=int, default=20)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--weight-decay", type=float, default=1e-4)
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--monitor", default="pixel_f1", choices=["pixel_f1", "pixel_iou", "image_auc"])
    t.add_argument("--out", default="runs/exp")
    return p.parse_args()


def run_epoch(model, loader, device, loss_fn, optimizer=None, scaler=None, scheduler=None):
    """One pass over ``loader``. Trains if ``optimizer`` is given, else evaluates."""
    train = optimizer is not None
    model.train(train)
    tracker = MetricTracker()
    total, n = 0.0, 0
    use_amp = device == "cuda"
    for x, y, label in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits.float(), y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            prob = torch.sigmoid(logits.float())
            tracker.update(prob, y, image_score(prob), label)
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / max(n, 1), (None if train else tracker)


def main() -> None:
    a = parse_args()
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"[train] device={device}")

    if a.dataset == "casia":
        if not a.data_root:
            raise SystemExit("--data-root is required for --dataset casia")
        samples = find_casia_v2(a.data_root)
    else:
        if not a.tampered_dir:
            raise SystemExit("--tampered-dir is required for --dataset folder")
        samples = find_folder_dataset(a.authentic_dir, a.tampered_dir, a.mask_dir)
    if a.limit:
        import random
        random.Random(a.seed).shuffle(samples)
        samples = samples[: a.limit]

    split = split_samples(samples, a.val_frac, a.test_frac, a.seed)
    save_split(split, out / "split.json")
    print("[train] split: " + ", ".join(f"{k}={len(v)}" for k, v in split.items()))

    ds_kw = dict(img_size=a.img_size, input_mode=a.input_mode, ela_quality=a.ela_quality, ela_scale=a.ela_scale)
    dl_kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=device == "cuda",
                 persistent_workers=a.workers > 0)
    train_dl = DataLoader(TamperDataset(split["train"], train=True, **ds_kw), shuffle=True, drop_last=True, **dl_kw)
    val_dl = DataLoader(TamperDataset(split["val"], **ds_kw), shuffle=False, **dl_kw)
    if len(train_dl) == 0:
        raise SystemExit("Training split is smaller than one batch; lower --batch-size or add data.")

    config = dict(arch=a.arch, encoder=a.encoder, input_mode=a.input_mode, img_size=a.img_size,
                  ela_quality=a.ela_quality, ela_scale=a.ela_scale)
    model = build_model(a.arch, a.encoder, in_channels(a.input_mode), pretrained=not a.no_pretrained).to(device)
    loss_fn = BCEDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps = a.epochs * len(train_dl)
    warmup = min(len(train_dl), steps // 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: (s + 1) / max(warmup, 1) if s < warmup
        else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(steps - warmup, 1))))
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")

    best, history = -1.0, []
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        train_loss, _ = run_epoch(model, train_dl, device, loss_fn, optimizer, scaler, scheduler)
        val_loss, tracker = run_epoch(model, val_dl, device, loss_fn)
        m = tracker.compute()
        m.update(epoch=epoch, train_loss=train_loss, val_loss=val_loss, seconds=round(time.time() - t0, 1))
        history.append(m)
        print(f"[epoch {epoch:02d}/{a.epochs}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"pixF1={m['pixel_f1']:.4f} IoU={m['pixel_iou']:.4f} "
              f"AUC={m.get('image_auc', float('nan')):.4f} acc={m['image_acc']:.4f} ({m['seconds']}s)")
        score = m.get(a.monitor, float("nan"))
        if not math.isnan(score) and score > best:
            best = score
            # Image-level threshold is calibrated on the validation set only.
            save_checkpoint(out / "best.pt", model, config, image_threshold=m["image_threshold"],
                            pixel_threshold=0.5, val_metrics=m, epoch=epoch)
            print(f"           saved best.pt ({a.monitor}={score:.4f}, image_threshold={m['image_threshold']:.3f})")
        (out / "history.json").write_text(json.dumps(history, indent=1))

    if best < 0:  # monitor never defined (e.g. no tampered images in val): keep the last model
        save_checkpoint(out / "best.pt", model, config, image_threshold=history[-1]["image_threshold"],
                        pixel_threshold=0.5, val_metrics=history[-1], epoch=a.epochs)
    print(f"[train] done. best {a.monitor}={best:.4f}. Next: python evaluate.py --checkpoint {out/'best.pt'} "
          f"--split {out/'split.json'}")


if __name__ == "__main__":
    main()
