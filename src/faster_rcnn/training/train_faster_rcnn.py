#!/usr/bin/env python3
"""
Train Faster R-CNN (polyp detection) using artifacts/train.json, val.json, roots_map.json.
"""

import argparse
import os
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

from faster_rcnn.augmentations.presets_faster_rcnn import (
    build_train_augs,
    build_val_augs,
)
from faster_rcnn.data.coco_detection_faster_rcnn import CocoDetDataset, collate_fn
from faster_rcnn.data.coco_eval_faster_rcnn import coco_map
from faster_rcnn.model_builder_faster_rcnn import build_model
from faster_rcnn.models.faster_rcnn_model import build_fasterrcnn
from faster_rcnn.training.engine_faster_rcnn import (
    save_checkpoint,
    train_one_epoch,
    validate_loss,
)
from faster_rcnn.training.optim_faster_rcnn import build_optimizer
from faster_rcnn.training.sched_faster_rcnn import build_scheduler
from faster_rcnn.utils.utils_faster_rcnn import (
    EarlyStopping,
    ensure_dir,
    set_seed,
    setup_perf,
)


# ARGS/CONFIGS
def get_parser():
    p = argparse.ArgumentParser("Train Faster R-CNN (polyp) with artifacts/ COCO")

    # data
    p.add_argument("--train-json", default="artifacts/train.json")
    p.add_argument("--val-json", default="artifacts/val.json")
    p.add_argument(
        "--roots-map",
        default="artifacts/roots_map.json",
        help="basename -> absolute image path (from prepare_data.py)",
    )
    # Optional fallback if some file_name entries are relative
    p.add_argument("--images-root", default=None)

    # Train params
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--img-size", type=int, default=832)
    p.add_argument(
        "--train-augs", choices=["none", "light", "medium", "strong"], default="light"
    )

    # Model
    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--freeze-backbone", type=int, default=0)
    p.add_argument("--no-pretrained", action="store_true")

    # Optimizer
    p.add_argument("--opt", choices=["sgd", "adam", "adamw"], default="sgd")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument(
        "--lr-backbone",
        type=float,
        default=None,
        help="If set, use this LR for backbone params (enables param groups)",
    )
    p.add_argument(
        "--lr-heads",
        type=float,
        default=None,
        help="If set, use this LR for heads (RPN+ROI) (enables param groups)",
    )
    p.add_argument(
        "--lr-tiered",
        action="store_true",
        help="If set, split backbone into body (0.5x) and FPN (1x) LRs plus heads",
    )

    # Scheduler
    p.add_argument(
        "--lr-scheduler",
        choices=["none", "step", "cosine", "plateau"],
        default="step",
    )
    p.add_argument("--step-size", type=int, default=5)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--cosine-tmax", type=int, default=10)
    p.add_argument(
        "--plateau-factor", type=float, default=0.1, help="LR multiply when plateau"
    )
    p.add_argument(
        "--plateau-patience",
        type=int,
        default=3,
        help="epochs with no improv before reduce",
    )
    p.add_argument(
        "--plateau-threshold",
        type=float,
        default=1e-3,
        help="min improvement to reset patience",
    )
    p.add_argument("--plateau-min-lr", type=float, default=1e-7)

    # Stopping
    p.add_argument("--early-patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=0.0)

    # Model

    # Device/Misc
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--tf32", action="store_true")

    # I/O
    p.add_argument("--out-dir", default="runs/frcnn_polyp")
    p.add_argument("--save-every", type=int, default=1)
    p.add_argument("--resume", default="")
    p.add_argument("--config", default="")

    return p


def apply_yaml(args):
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f) or {}
        for k, v in cfg.items():
            if hasattr(args, k):
                setattr(args, k, v)
    return args


# DATALOADERS


def build_loaders(args):
    train_tf = build_train_augs(args.img_size, args.train_augs)
    val_tf = build_val_augs(args.img_size)

    train_ds = CocoDetDataset(
        ann_file=args.train_json,
        roots_map_path=args.roots_map,
        images_root=args.images_root,
        transform=train_tf,
    )

    val_ds = CocoDetDataset(
        ann_file=args.val_json,
        roots_map_path=args.roots_map,
        images_root=args.images_root,
        transform=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )
    return train_loader, val_loader


def maybe_resume(args, model, optimizer, scheduler, scaler):
    start_epoch, best_val = 1, float("inf")
    if not args.resume or not os.path.isfile(args.resume):
        return start_epoch, best_val

    ckpt = torch.load(args.resume, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler and "scheduler" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler"])
        except Exception:
            print("[warn] scheduler state mismatch; continuing.")
    if "scaler" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler"])
        except Exception:
            print("[warn] scaler state mismatch; continuing.")
    start_epoch = ckpt["epoch"] + 1
    best_val = ckpt.get("best_val", best_val)
    print(f"[resume] {args.resume} @ epoch {start_epoch-1} (best_val={best_val:.4f})")
    return start_epoch, best_val


def main():
    args = apply_yaml(get_parser().parse_args())

    # setup
    set_seed(args.seed, deterministic=args.deterministic)
    setup_perf(tf32=args.tf32)

    train_loader, val_loader = build_loaders(args)

    # Model
    model, device = build_model(
        num_classes=args.num_classes,
        pretrained=(not args.no_pretrained),
        freeze_backbone=args.freeze_backbone,
        device=args.device,
    )

    # Optimizer
    optimizer = build_optimizer(
        model,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        lr_backbone=getattr(args, "lr_backbone", None),
        lr_heads=getattr(args, "lr_heads", None),
        tiered=getattr(args, "lr_tiered", False),
    )

    scheduler = build_scheduler(
        args.lr_scheduler,
        optimizer,
        step_size=args.step_size,
        gamma=args.gamma,
        cosine_tmax=args.cosine_tmax,
        plateau_factor=args.plateau_factor,
        plateau_patience=args.plateau_patience,
        plateau_threshold=args.plateau_threshold,
        plateau_min_lr=args.plateau_min_lr,
    )

    # AMP + Early stopping
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    start_epoch, best_val = maybe_resume(args, model, optimizer, scheduler, scaler)

    ensure_dir(args.out_dir)
    stopper = EarlyStopping(patience=args.early_patience, min_delta=args.min_delta)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch
        )
        val_loss = validate_loss(model, val_loader, device)

        # scheduler step
        if scheduler:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)  # driven by val metric
            else:
                scheduler.step()

        curr_lr = optimizer.param_groups[0]["lr"]
        print(
            f"[Epoch {epoch:03d}] lr={curr_lr:.6f} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}"
        )

        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss

        # Save checkpoints
        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler else {},
            "scaler": scaler.state_dict(),
            "best_val": best_val,
            "args": vars(args),
        }

        # save
        save_checkpoint(state, args.out_dir, is_best=False, filename="last.pth")
        if (epoch % args.save_every) == 0:
            save_checkpoint(
                state, args.out_dir, is_best=False, filename=f"epoch_{epoch:03d}.pth"
            )
        if is_best:
            save_checkpoint(state, args.out_dir, is_best=True, filename="best.pth")

        if stopper.step(val_loss):
            print("[early-stop] stopping.")
            break

    print(
        f"[done] best val_loss={best_val:.4f} ; best checkpoint at {args.out_dir}/best.pth"
    )


if __name__ == "__main__":
    main()
