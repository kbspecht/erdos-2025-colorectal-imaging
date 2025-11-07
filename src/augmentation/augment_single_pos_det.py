#!/usr/bin/env python3
"""
augment_single_pos_det.py
-------------------------
Static Albumentations augmentations for single-image polyp **detection** data.

This script:
  1) Loads each image and its corresponding bbox .txt (absolute pixels, pascal_voc style lines):
        <class_name_or_id> x_min y_min x_max y_max
  2) Applies Albumentations augmentations jointly (image + bboxes).
  3) Saves augmented images and YOLO **detection** labels:
        class_id cx_norm cy_norm w_norm h_norm

Typical usage:
!python ../src/augment_single_pos_det.py \
  --img_root   "../data/detection2/train/images_single" \
  --bbox_root  "../data/detection2/train/bbox" \
  --out_images "../data/detection2/aug_pos_images_det" \
  --out_labels "../data/detection2/aug_pos_labels_det" \
  --copies_per_img 2 \
  --class_map "polyp:0" \
  --min_area_rel 0.001 --min_w_rel 0.004 --min_h_rel 0.004 \
  --min_w_px 4 --min_h_px 4 \
  --seed 0

Notes:
- Exact class mapping is configurable; numeric IDs are respected if present.
- Uses bbox-safe crops; also prunes tiny boxes post-augmentation.
- By default, aug samples with 0 remaining boxes are skipped (see --skip_if_no_boxes).
"""

from __future__ import annotations
import argparse, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ------------------------------ Parsing ---------------------------------- #

def parse_class_map(spec: Optional[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if not spec:
        return mapping
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Invalid --class_map entry: '{part}' (expected 'name:id').")
        name, id_str = part.split(":", 1)
        name = name.strip()
        id_str = id_str.strip()
        if not name or not id_str.isdigit():
            raise ValueError(f"Invalid --class_map pair: '{part}'.")
        mapping[name] = int(id_str)
    return mapping

def read_pascal_bboxes(txt_path: Path, class_map: Dict[str,int], default_class_id: int) -> List[Tuple[int,float,float,float,float]]:
    """Read lines like '<cls> x1 y1 x2 y2' (pixels)."""
    if not txt_path.exists():
        return []
    out = []
    for ln in txt_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        ln = ln.strip().replace(",", " ")
        if not ln:
            continue
        parts = [p for p in ln.split() if p]
        if len(parts) < 5:
            continue
        ctoken = parts[0]
        if ctoken.isdigit():
            cid = int(ctoken)
        else:
            cid = class_map.get(ctoken, default_class_id)
        try:
            x1, y1, x2, y2 = map(float, parts[1:5])
        except Exception:
            continue
        out.append((cid, x1, y1, x2, y2))
    return out

# -------------------------- Geometry helpers ----------------------------- #

def clip_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float,float,float,float]:
    x1c = min(max(x1, 0.0), max(W - 1.0, 0.0))
    y1c = min(max(y1, 0.0), max(H - 1.0, 0.0))
    x2c = min(max(x2, 0.0), max(W - 1.0, 0.0))
    y2c = min(max(y2, 0.0), max(H - 1.0, 0.0))
    xl, xr = min(x1c, x2c), max(x1c, x2c)
    yt, yb = min(y1c, y2c), max(y1c, y2c)
    return xl, yt, xr, yb

def to_yolo_norm(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float,float,float,float]:
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

def passes_filters(x1: float, y1: float, x2: float, y2: float, W: int, H: int,
                   min_area_rel: float, min_w_rel: float, min_h_rel: float,
                   min_w_px: int, min_h_px: int) -> bool:
    w = x2 - x1; h = y2 - y1
    if w <= 0 or h <= 0:
        return False
    area_rel = (w * h) / float(max(W * H, 1))
    if area_rel < min_area_rel:
        return False
    if (w / W) < min_w_rel or (h / H) < min_h_rel:
        return False
    if w < min_w_px or h < min_h_px:
        return False
    return True

# --------------------------- Aug definitions ----------------------------- #

def build_aug(H: int, W: int) -> A.BasicTransform:
    """
    Builds a bbox-safe augmentation pipeline.
    Uses adaptive target sizes based on the input image (like your seg script).
    """
    if min(H, W) < 352:
        crop_h, crop_w = 256, 256
    elif min(H, W) < 512:
        crop_h, crop_w = 352, 352
    else:
        crop_h, crop_w = 480, 480

    return A.Compose(
        [
            A.PadIfNeeded(min_height=crop_h, min_width=crop_w,
                          border_mode=cv2.BORDER_CONSTANT, value=0, p=1.0),
            # Bbox-safe crop that tries hard to keep boxes visible
            A.RandomSizedBBoxSafeCrop(height=crop_h, width=crop_w,
                                      erosion_rate=0.2, p=0.9),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.20, rotate_limit=20,
                               border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.OneOf([
                A.RandomBrightnessContrast(0.25, 0.25, p=1.0),
                A.HueSaturationValue(5, 10, 10, p=1.0),
                A.RandomGamma((80, 120), p=1.0),
            ], p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=3, p=1.0),
                A.GaussNoise(var_limit=(5, 25), p=1.0),
            ], p=0.25),
            A.HorizontalFlip(p=0.5),
        ],
        bbox_params=A.BboxParams(
            format="pascal_voc",            # [x_min, y_min, x_max, y_max]
            min_visibility=0.30,            # drop boxes that are mostly out
            label_fields=["class_labels"],  # tie boxes to these labels
        ),
    )

# ------------------------------ Core ------------------------------------- #

def main(args):
    random.seed(args.seed); np.random.seed(args.seed)

    img_root   = Path(args.img_root)
    bbox_root  = Path(args.bbox_root)
    out_images = Path(args.out_images)
    out_labels = Path(args.out_labels)
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    class_map = parse_class_map(args.class_map)

    img_files = [p for p in sorted(img_root.iterdir()) if p.suffix.lower() in IMG_EXTS]
    print(f"[info] images found: {len(img_files)}")

    total_augs   = 0
    total_kept   = 0
    total_images = len(img_files)

    for i, img_path in enumerate(img_files, 1):
        stem = img_path.stem
        ann  = bbox_root / f"{stem}.txt"

        im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if im is None:
            if args.verbose:
                print(f"[warn] failed to read image: {img_path}")
            continue

        H, W = im.shape[:2]
        # read original boxes
        raw_boxes = read_pascal_bboxes(ann, class_map, args.default_class_id)

        # Build inputs for albumentations: bboxes list and aligned class label list
        bboxes = [[x1, y1, x2, y2] for (_, x1, y1, x2, y2) in raw_boxes]
        labels = [cid for (cid, *_coords) in raw_boxes]

        aug = build_aug(H, W)

        for c in range(args.copies_per_img):
            # If no boxes exist, still allow augmentation (will usually be dropped by min_visibility)
            out = aug(image=im, bboxes=bboxes, class_labels=labels)
            aug_img   = out["image"]
            aug_boxes = out["bboxes"]
            aug_lbls  = out["class_labels"]
            H2, W2 = aug_img.shape[:2]

            # Post-filter & clip
            yolo_lines: List[str] = []
            kept = 0
            for (cid, (x1, y1, x2, y2)) in zip(aug_lbls, aug_boxes):
                x1c, y1c, x2c, y2c = clip_box(x1, y1, x2, y2, W2, H2)
                if not passes_filters(
                    x1c, y1c, x2c, y2c, W2, H2,
                    min_area_rel=args.min_area_rel,
                    min_w_rel=args.min_w_rel,
                    min_h_rel=args.min_h_rel,
                    min_w_px=args.min_w_px,
                    min_h_px=args.min_h_px,
                ):
                    if args.verbose > 1:
                        print(f"[skip] tiny/noisy ({stem}) -> ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                    continue
                cx, cy, w, h = to_yolo_norm(x1c, y1c, x2c, y2c, W2, H2)
                yolo_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                kept += 1

            # Handle zero-box outcomes
            if kept == 0 and args.skip_if_no_boxes:
                # skip saving this augmentation if nothing usable remains
                continue

            aug_name = f"{stem}_aug{c+1:02d}"
            img_out_path = out_images / f"{aug_name}.jpg"
            lbl_out_path = out_labels / f"{aug_name}.txt"

            cv2.imwrite(str(img_out_path), aug_img)
            lbl_out_path.write_text("\n".join(yolo_lines))

            total_augs += 1
            total_kept += kept

        if (i % 100 == 0 or i == total_images) and args.verbose:
            print(f"[progress] {i}/{total_images} | aug_written={total_augs} | boxes_kept={total_kept}")

    print(f"[done] images={total_images} | aug_written={total_augs} | boxes_kept={total_kept} | out_images={out_images} | out_labels={out_labels}")

# ------------------------------- CLI ------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--bbox_root", type=str, required=True)
    ap.add_argument("--out_images", type=str, required=True)
    ap.add_argument("--out_labels", type=str, required=True)
    ap.add_argument("--copies_per_img", type=int, default=2)

    # class mapping / defaults
    ap.add_argument("--class_map", type=str, default="polyp:0")
    ap.add_argument("--default_class_id", type=int, default=0)

    # tiny/noise filters (post-aug)
    ap.add_argument("--min_area_rel", type=float, default=0.001)
    ap.add_argument("--min_w_rel",   type=float, default=0.004)
    ap.add_argument("--min_h_rel",   type=float, default=0.004)
    ap.add_argument("--min_w_px",    type=int,   default=4)
    ap.add_argument("--min_h_px",    type=int,   default=4)

    ap.add_argument("--skip_if_no_boxes", action="store_true", default=True,
                    help="If set (default), skip saving an aug sample when 0 boxes remain.")

    # reproducibility / verbosity
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--verbose", type=int, default=1, help="0=silent, 1=progress, 2=include [skip] lines")

    args = ap.parse_args()
    main(args)
