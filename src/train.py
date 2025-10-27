# src/train.py
"""
Minimal, production-friendly training entrypoint for Ultralytics YOLO (v8/v11).

- Argparse CLI
- Optional YAML to override/add Ultralytics train kwargs (optimizer, lr0, patience, cos_lr, cache, etc.)
- Albumentations pipeline from YAML (applied via on_preprocess_batch)
- Writes run metadata (seed, git commit, env) next to Ultralytics artifacts
- Explicit final validation on best weights
- Works with yolov8 / yolo11 via the 'ultralytics' pip package

Usage (from repo root):
  python -m src.train --model yolov8s.pt --data configs/data_polyp.yaml --cfg configs/polyp_alb352_cos.yaml
"""

from __future__ import annotations
import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml
import numpy as np
import torch
import albumentations as A
from albumentations.core.serialization import from_dict
from ultralytics import YOLO


def _load_yaml(path: str | Path | None) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return out or None
    except Exception:
        return None


def _env_info() -> Dict[str, Any]:
    """Collect lightweight environment info for reproducibility."""
    try:
        torch_cuda = {
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "current_device": torch.cuda.current_device() if torch.cuda.is_available() else None,
        }
    except Exception:
        torch_cuda = {"available": None, "device_count": None, "current_device": None}

    try:
        import ultralytics
        u_version = ultralytics.__version__
    except Exception:
        u_version = None

    return {
        "python": os.environ.get("PYTHON_VERSION"),
        "ultralytics_version": u_version,
        "torch_cuda": torch_cuda,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train YOLO (Ultralytics) with a reproducible CLI")
    ap.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or path, e.g. yolo11s.pt (finetune) or yolo11n.yaml (from scratch)",
    )
    ap.add_argument("--data", type=str, required=True, help="Dataset YAML (paths, nc, names)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default="0", help="'cpu' or CUDA like '0' or '0,1'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=str, default="runs/train", help="Base directory for runs")
    ap.add_argument("--name", type=str, default="exp", help="Run name (folder under --project)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--single-cls", action="store_true", help="Treat dataset as single class")
    ap.add_argument("--resume", action="store_true", help="Resume last run in this project/name")
    ap.add_argument(
        "--cfg",
        type=str,
        default=None,
        help="Optional YAML with extra train kwargs (put optimizer etc. under `optim:` and Albumentations under `albumentations:`)",
    )
    ap.add_argument("--notes", type=str, default="", help="Freeform notes stored with the run")
    ap.add_argument("--lr0", type=float, default=None, help="Initial learning rate (overrides cfg)")
    return ap.parse_args()


def main() -> None:
    opt = parse_args()

    save_dir = Path(opt.project) / opt.name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load optional extra kwargs for model.train(...)
    cfg_dict = _load_yaml(opt.cfg)

    # Sub-dicts: keep these OUT of Ultralytics kwargs unless intended
    optim_kwargs = cfg_dict.get("optim", {}) if cfg_dict else {}
    alb_cfg = cfg_dict.get("albumentations") if cfg_dict else None

    # ---- Build Albumentations transform (if present) ----
    alb_transform = None
    if alb_cfg and "transform" in alb_cfg:
        composed = from_dict({"transform": alb_cfg["transform"]})
        # Recompose with bbox params suitable for YOLO (normalized xywh)
        alb_transform = A.Compose(
            composed.transforms,
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.05,        # drop boxes mostly cropped out
            ),
        )

    # ---- Build or load model ----
    model = YOLO(opt.model)

    # ---- Assemble train kwargs explicitly ----
    train_kwargs: Dict[str, Any] = dict(
        data=opt.data,
        epochs=opt.epochs,
        imgsz=opt.imgsz,
        batch=opt.batch,
        device=opt.device,
        workers=opt.workers,
        project=str(opt.project),
        name=opt.name,
        seed=opt.seed,
        single_cls=opt.single_cls,
        resume=opt.resume,
        verbose=True,
    )

    # Only pass optimizer/schedule/etc. (avoid dumping whole cfg_dict)
    if optim_kwargs:
        train_kwargs.update(optim_kwargs)

    # CLI --lr0 overrides YAML if provided
    if opt.lr0 is not None:
        train_kwargs["lr0"] = opt.lr0

    # ---- Register Albumentations preprocessing callback ----
    if alb_transform:
        def _alb_on_preprocess_batch(trainer, batch):
            imgs = batch["img"]  # (B, 3, H, W), float32 in [0,1]
            nb = imgs.shape[0]
            new_imgs, new_boxes, new_cls = [], [], []

            # batch["bboxes"] and batch["cls"] are lists (len B) in recent Ultralytics
            bboxes_list = batch["bboxes"]
            cls_list = batch["cls"]

            for i in range(nb):
                # HWC uint8 for Albumentations
                im = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

                # Extract per-image boxes and labels
                b = bboxes_list[i]
                c = cls_list[i]

                if torch.is_tensor(b):
                    b = b.cpu().numpy()
                if torch.is_tensor(c):
                    c = c.cpu().numpy()

                b_list = b.tolist() if b is not None and len(b) else []
                class_labels = [int(x) for x in c.reshape(-1).tolist()] if c is not None and len(c) else []

                aug = alb_transform(image=im, bboxes=b_list, class_labels=class_labels)
                im2 = aug["image"]
                bb2 = aug["bboxes"]
                cl2 = aug["class_labels"]

                # back to tensors
                im2_t = torch.from_numpy(im2).permute(2, 0, 1).to(imgs.device).float() / 255.0
                new_imgs.append(im2_t)

                if len(bb2):
                    new_boxes.append(torch.tensor(bb2, device=imgs.device, dtype=torch.float32))
                    new_cls.append(torch.tensor(cl2, device=imgs.device, dtype=torch.float32).unsqueeze(-1))
                else:
                    new_boxes.append(torch.zeros((0, 4), device=imgs.device, dtype=torch.float32))
                    new_cls.append(torch.zeros((0, 1), device=imgs.device, dtype=torch.float32))

            batch["img"] = torch.stack(new_imgs)
            batch["bboxes"] = new_boxes
            batch["cls"] = new_cls
            return batch

        model.add_callback("on_preprocess_batch", _alb_on_preprocess_batch)

    # ---- Kick off training ----
    results = model.train(**train_kwargs)

    # ---- Validate on best checkpoint (explicit final val) ----
    model.val(data=opt.data, imgsz=opt.imgsz, device=opt.device)

    # ---- Save lightweight run metadata for reproducibility ----
    meta = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "args": vars(opt),
        "extra_train_kwargs": cfg_dict,
        "merged_train_kwargs": train_kwargs,
        "git_commit": _git_commit(),
        "env": _env_info(),
        "artifacts_dir": str(save_dir.resolve()),
        "notes": opt.notes,
    }
    with (save_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(
        f"\n✅ Finished. Artifacts saved under: {save_dir}\n"
        f"   - weights/: best.pt, last.pt\n"
        f"   - results.csv / results.png\n"
        f"   - run_meta.json (seed, git, env, merged kwargs)\n"
    )


if __name__ == "__main__":
    main()
