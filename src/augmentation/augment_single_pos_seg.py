"""
augment_single_pos_seg.py
-------------------------
Static Albumentations augmentations for single-image polyp segmentation data.

This script:
  1. Loads each image and its corresponding mask.
  2. Applies Albumentations augmentations jointly (image + mask).
  3. Saves augmented images and masks.
  4. Extracts polygons from augmented masks → writes YOLO segmentation labels.

Usage:
!python ../src/augment_single_pos_seg.py \
  --img_root "../data/detection2/train/images_single" \
  --mask_roots "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C1" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C2/masks_C2" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C3/masks_C3" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C4/masks_C4" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C5/masks_C5" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C6/masks_C6" \
  --out_images "../data/segmentation2/aug_pos_images" \
  --out_masks  "../data/segmentation2/aug_pos_masks" \
  --out_labels "../data/segmentation2/aug_pos_labels_seg" \
  --copies_per_img 2 \
  --approx_eps 1.5 \
  --min_area_px 25 \
  --seed 0
"""

import argparse
from pathlib import Path
import random
import cv2
import numpy as np
import albumentations as A

# --------------------------------------------------------------
# Polygon extraction helpers (same as in masks_to_yolo_seg_labels)
# --------------------------------------------------------------
def mask_to_polygons(mask, approx_eps=1.5, min_area_px=25):
    mask = mask.astype(np.uint8)
    _, binm = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < float(min_area_px):
            continue
        if approx_eps > 0:
            c = cv2.approxPolyDP(c, epsilon=approx_eps, closed=True)
        c = c.reshape(-1, 2).astype(np.float32)
        if c.shape[0] >= 3:
            polys.append(c)
    return polys


def normalize_polygon(poly, W, H):
    poly = poly.astype(np.float32)
    poly[:, 0] = np.clip(poly[:, 0] / W, 0, 1)
    poly[:, 1] = np.clip(poly[:, 1] / H, 0, 1)
    return poly


def polygon_to_yolo_line(poly_norm, class_id=0):
    coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly_norm])
    return f"{class_id} {coords}"


# --------------------------------------------------------------
# Mask lookup (handles _mask or no suffix)
# --------------------------------------------------------------
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
def find_mask_for_image(stem, mask_roots):
    candidates = []
    for root in mask_roots:
        for ext in MASK_EXTS:
            p1 = root / f"{stem}_mask{ext}"
            p2 = root / f"{stem}{ext}"
            if p1.exists():
                candidates.append(p1)
            elif p2.exists():
                candidates.append(p2)
    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda p: (p.suffix.lower() != ".png", str(p)))
    return candidates[0]


# --------------------------------------------------------------
# Main augmentation logic
# --------------------------------------------------------------
def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    img_root = Path(args.img_root)
    mask_roots = [Path(p) for p in args.mask_roots]
    out_images = Path(args.out_images)
    out_masks = Path(args.out_masks)
    out_labels = Path(args.out_labels)

    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    img_files = [p for p in sorted(img_root.iterdir()) if p.suffix.lower() in MASK_EXTS]
    print(f"[info] images found: {len(img_files)}")

    total_written = 0

    for i, img_path in enumerate(img_files, 1):
        stem = img_path.stem
        mask_path = find_mask_for_image(stem, mask_roots)
        if mask_path is None:
            continue

        im = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if im is None or mask is None:
            continue

        H, W = im.shape[:2]

        # adaptive crop sizes
        if min(H, W) < 352:
            crop_h, crop_w = 256, 256
        elif min(H, W) < 512:
            crop_h, crop_w = 352, 352
        else:
            crop_h, crop_w = 480, 480

        # define augmentation per image
        aug = A.Compose([
            A.PadIfNeeded(min_height=crop_h, min_width=crop_w,
                          border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=1.0),
            A.CropNonEmptyMaskIfExists(height=crop_h, width=crop_w, p=0.95),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.20, rotate_limit=20,
                               border_mode=cv2.BORDER_CONSTANT, p=0.5),
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
        ])

        for c in range(args.copies_per_img):
            out = aug(image=im, mask=mask)
            aug_img, aug_mask = out["image"], out["mask"]
            H2, W2 = aug_img.shape[:2]

            aug_name = f"{stem}_aug{c+1:02d}"
            img_out_path = out_images / f"{aug_name}.jpg"
            mask_out_path = out_masks / f"{aug_name}.png"
            label_out_path = out_labels / f"{aug_name}.txt"

            cv2.imwrite(str(img_out_path), aug_img)
            cv2.imwrite(str(mask_out_path), aug_mask)

            # Generate YOLO-seg label
            polys = mask_to_polygons(aug_mask, approx_eps=args.approx_eps, min_area_px=args.min_area_px)
            lines = []
            for poly in polys:
                poly_norm = normalize_polygon(poly, W2, H2)
                line = polygon_to_yolo_line(poly_norm, class_id=0)
                lines.append(line)

            label_out_path.write_text("\n".join(lines))
            total_written += 1

        if i % 100 == 0 or i == len(img_files):
            print(f"[progress] {i}/{len(img_files)} images processed, total_aug={total_written}")

    print(f"[done] Augmented images={total_written} written to {out_images}")


# --------------------------------------------------------------
# CLI
# --------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True)
    ap.add_argument("--mask_roots", type=str, nargs="+", required=True)
    ap.add_argument("--out_images", type=str, required=True)
    ap.add_argument("--out_masks", type=str, required=True)
    ap.add_argument("--out_labels", type=str, required=True)
    ap.add_argument("--copies_per_img", type=int, default=2)
    ap.add_argument("--approx_eps", type=float, default=1.5)
    ap.add_argument("--min_area_px", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(args)
