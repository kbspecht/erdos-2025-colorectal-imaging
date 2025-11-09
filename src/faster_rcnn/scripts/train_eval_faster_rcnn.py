#!/usr/bin/env python3
"""
End-to-end Faster R-CNN pipeline for colorectal polyp detection.

Steps:
1. (Optionally) prepare data → src/artifacts/{train.json, val.json, roots_map.json}
2. Train Faster R-CNN via existing CLI (faster_rcnn.training.train_faster_rcnn)
3. Copy a chosen checkpoint into models/faster_rcnn with a unique name.
4. Run COCO mAP evaluation on the validation set and print metrics.

Designed to be CI/CD-friendly and avoid overwriting existing checkpoints
(e.g., frcnn_imgsz832f0_best.pth).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch

# ----------------------------------------------------------------------
# Paths
# This file: repo_root/src/faster_rcnn/scripts/train_eval_faster_rcnn.py
# parents[0] = scripts
# parents[1] = faster_rcnn
# parents[2] = src          -> SRC_ROOT
# parents[3] = repo root    -> REPO_ROOT
# ----------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = SRC_ROOT.parent
MODELS_DIR = REPO_ROOT / "models" / "faster_rcnn"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Ensure src is importable
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from faster_rcnn.data.coco_eval_faster_rcnn import coco_map
from faster_rcnn.data.data_loaders_faster_rcnn import build_train_val_loaders
from faster_rcnn.models.faster_rcnn_model import build_fasterrcnn


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def pick_checkpoint(checkpoints_dir: Path, epochs: int | None = None) -> Path:
    """
    Choose which checkpoint to use, without hard-coding a specific filename.

    Priority:
      1. epoch_{epochs:03d}.pth in checkpoints_dir if epochs given and exists
      2. Latest *.pth file in checkpoints_dir (by modification time)
    """
    if epochs is not None:
        ep = checkpoints_dir / f"epoch_{epochs:03d}.pth"
        if ep.exists():
            return ep

    ckpts = sorted(
        checkpoints_dir.glob("*.pth"),
        key=lambda p: p.stat().st_mtime,
    )
    if not ckpts:
        raise FileNotFoundError(f"No .pth checkpoints found in {checkpoints_dir}")
    return ckpts[-1]


def make_unique_dest(dir_: Path, base_name: str) -> Path:
    """
    Create a unique path in dir_ based on base_name, without overwriting existing files.
    Example:
        base_name = "epoch_001.pth"
        if exists:
            epoch_001_run1.pth, epoch_001_run2.pth, ...
    """
    candidate = dir_ / base_name
    if not candidate.exists():
        return candidate

    stem = candidate.stem  # "epoch_001"
    suffix = candidate.suffix  # ".pth"

    for k in range(1, 1000):
        alt = dir_ / f"{stem}_run{k}{suffix}"
        if not alt.exists():
            return alt

    raise RuntimeError(f"Could not create unique checkpoint name for {base_name}")


def run_prepare_data(datasets: Path, out_dir: Path, class_name: str = "polyp") -> None:
    """
    Call your existing prepare_data_faster_rcnn script via `python -m`.
    """
    cmd = [
        sys.executable,
        "-m",
        "faster_rcnn.data.prepare_data_faster_rcnn",
        "--datasets",
        str(datasets),
        "--out-dir",
        str(out_dir),
        "--class-name",
        class_name,
    ]
    print(f"[info] Running data prep: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def run_training(
    train_json: Path,
    val_json: Path,
    roots_map: Path,
    images_root: Path,
    epochs: int = 1,
) -> Path:
    """
    Call your existing train_faster_rcnn CLI.

    Assumes the training script writes checkpoints under:
        repo_root/runs/faster_rcnn/

    Returns:
        Path to the *copied* checkpoint under models/faster_rcnn (unique name).
    """
    runs_dir = REPO_ROOT / "runs" / "faster_rcnn"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "-m",
        "faster_rcnn.training.train_faster_rcnn",
        "--train-json",
        str(train_json),
        "--val-json",
        str(val_json),
        "--roots-map",
        str(roots_map),
        "--images-root",
        str(images_root),
        "--epochs",
        str(epochs),
        "--batch-size",
        "8",
        "--num-workers",
        "0",
        "--img-size",
        "640",
        "--train-augs",
        "medium",
        "--num-classes",
        "2",
        "--freeze-backbone",
        "2",
        "--opt",
        "sgd",
        "--lr-backbone",
        "1e-4",
        "--lr-heads",
        "5e-3",
        "--weight-decay",
        "1e-4",
        "--lr-scheduler",
        "none",
        "--device",
        "cpu",
    ]
    print(f"[info] Running training: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    # Pick a checkpoint from runs/frcnn_polyp
    raw_ckpt = pick_checkpoint(runs_dir, epochs=epochs)
    print(f"[info] Selected raw checkpoint: {raw_ckpt}")

    # Copy into models/faster_rcnn with a unique name (never overwrites existing files)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest_ckpt = make_unique_dest(MODELS_DIR, raw_ckpt.name)
    shutil.copy2(raw_ckpt, dest_ckpt)
    print(f"[info] Copied checkpoint to: {dest_ckpt}")

    return dest_ckpt


def run_eval(
    ckpt_path: Path,
    train_json: Path,
    val_json: Path,
    roots_map: Path,
    images_root: Path,
) -> None:
    """
    Run COCO mAP evaluation (almost exactly what you had in the notebook).
    """
    device = torch.device("cpu")  # CI-friendly, override if needed

    print(f"[info] Loading model from: {ckpt_path}")
    model = build_fasterrcnn(num_classes=2, pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    state_dict = (
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    model.load_state_dict(state_dict)
    model.to(device).eval()

    print("[info] Building val loader...")
    _, val_loader = build_train_val_loaders(
        train_json=str(train_json),
        val_json=str(val_json),
        roots_map=str(roots_map),
        images_root=str(images_root),
        img_size=832,
        train_augs="light",
        batch_size=2,
        num_workers=0,
    )

    print("[info] Running COCO eval...")
    metrics = coco_map(
        model,
        val_loader,
        device,
        coco_gt_json=str(val_json),
        verbose=True,
        log_every=1,
    )

    print("\nFinal metrics:")
    print(
        f"  mAP@[.5:.95] = {metrics['mAP_50_95']:.4f}\n"
        f"  mAP@0.5      = {metrics['mAP_50']:.4f}\n"
        f"  Recall       = {metrics['recall']:.4f}\n"
        f"  Precision    = {metrics['precision']:.4f}"
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Train + eval Faster R-CNN polyp model.")
    p.add_argument(
        "--datasets",
        type=Path,
        default=REPO_ROOT / "testing_faster",
        help="Raw dataset root for prepare_data_faster_rcnn.",
    )
    p.add_argument(
        "--images-root",
        type=Path,
        required=True,
        help="Images root used by train_faster_rcnn and data loaders.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=Path,
        default=SRC_ROOT / "artifacts",
        help="Where train.json / val.json / roots_map.json will be stored.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train for.",
    )
    p.add_argument(
        "--skip-prepare",
        action="store_true",
        help="Skip data prep if artifacts already exist.",
    )
    args = p.parse_args()

    train_json = args.artifacts_dir / "train.json"
    val_json = args.artifacts_dir / "val.json"
    roots_map = args.artifacts_dir / "roots_map.json"

    if not args.skip_prepare or not (train_json.exists() and val_json.exists()):
        run_prepare_data(args.datasets, args.artifacts_dir, class_name="polyp")
    else:
        print("[info] Skipping data prep (artifacts already exist).")

    ckpt_path = run_training(
        train_json=train_json,
        val_json=val_json,
        roots_map=roots_map,
        images_root=args.images_root,
        epochs=args.epochs,
    )

    run_eval(
        ckpt_path=ckpt_path,
        train_json=train_json,
        val_json=val_json,
        roots_map=roots_map,
        images_root=args.images_root,
    )


if __name__ == "__main__":
    main()
