#!/usr/bin/env python3
"""
bboxes_to_yolo_det_labels.py  (with tiny/noise box filtering)
----------------------------------------------------------------
Reads per-image bbox txts of the form:
    <class_name_or_id> x_min y_min x_max y_max   # pixel coords
and writes YOLO detection labels:
    class_id x_center_norm y_center_norm w_norm h_norm

Now with filters to drop unreasonable boxes:
- min relative area, width, height
- min absolute pixel width/height
"""

from __future__ import annotations
from pathlib import Path
import argparse
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# ------------------------------- Parsing ---------------------------------- #

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


def parse_bbox_line(
    line: str,
    class_map: Dict[str, int],
    default_class_id: int = 0
) -> Optional[Tuple[int, float, float, float, float]]:
    line = line.strip().replace(",", " ")
    if not line:
        return None
    parts = [p for p in line.split() if p]
    if len(parts) < 5:
        return None
    cls_token = parts[0]
    coords = parts[1:5]

    if cls_token.isdigit():
        cls_id = int(cls_token)
    else:
        cls_id = class_map.get(cls_token, default_class_id)

    try:
        x1, y1, x2, y2 = map(float, coords)
    except Exception:
        return None
    return cls_id, x1, y1, x2, y2

# --------------------------- Geometry helpers ----------------------------- #

def clip_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    x1c = min(max(x1, 0.0), W - 1.0)
    y1c = min(max(y1, 0.0), H - 1.0)
    x2c = min(max(x2, 0.0), W - 1.0)
    y2c = min(max(y2, 0.0), H - 1.0)
    x_left  = min(x1c, x2c)
    x_right = max(x1c, x2c)
    y_top   = min(y1c, y2c)
    y_bot   = max(y1c, y2c)
    return x_left, y_top, x_right, y_bot

def box_to_yolo_norm(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float, float, float, float]:
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    if W <= 0 or H <= 0:
        raise ValueError("Invalid image size.")
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

# ----------------------------- IO helpers --------------------------------- #

def write_label_file(lbl_path: Path, lines: List[str]) -> None:
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.write_text("\n".join(lines))

def find_images(img_root: Path) -> List[Path]:
    files: List[Path] = []
    for p in img_root.rglob("*"):
        if p.suffix.lower() in IMG_EXTS and p.is_file():
            files.append(p)
    files.sort()
    return files

# ------------------------------ Core logic -------------------------------- #

def passes_filters(
    x1: float, y1: float, x2: float, y2: float,
    W: int, H: int,
    min_area_rel: float,
    min_w_rel: float, min_h_rel: float,
    min_w_px: int, min_h_px: int,
) -> bool:
    w_px = x2 - x1
    h_px = y2 - y1
    if w_px <= 0 or h_px <= 0:
        return False
    area_rel = (w_px * h_px) / float(W * H)
    if area_rel < min_area_rel:
        return False
    if (w_px / W) < min_w_rel or (h_px / H) < min_h_rel:
        return False
    if w_px < min_w_px or h_px < min_h_px:
        return False
    return True

def process_dataset(
    img_root: Path,
    bbox_root: Path,
    out_labels: Path,
    class_map: Dict[str, int],
    default_class_id: int = 0,
    verbose: bool = True,
    skip_empty: bool = False,
    warn_on_missing_ann: bool = True,
    min_area_rel: float = 5e-4,   # 0.05% of image area
    min_w_rel: float = 0.005,     # 0.5% of width
    min_h_rel: float = 0.005,     # 0.5% of height
    min_w_px: int = 4,
    min_h_px: int = 4,
) -> Tuple[int, int, int]:
    images = find_images(img_root)
    n_imgs = len(images)
    n_with = 0
    n_boxes = 0

    for i, ip in enumerate(images, 1):
        stem = ip.stem
        lblp = out_labels / f"{stem}.txt"
        annp = bbox_root / f"{stem}.txt"

        im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if im is None:
            if verbose:
                print(f"[warn] failed to read image: {ip}")
            if not skip_empty:
                write_label_file(lblp, [])
            continue
        H, W = im.shape[:2]

        lines_out: List[str] = []

        if annp.exists():
            raw = annp.read_text(encoding="utf-8", errors="ignore").splitlines()
            for k, line in enumerate(raw, 1):
                parsed = parse_bbox_line(line, class_map=class_map, default_class_id=default_class_id)
                if parsed is None:
                    continue
                cls_id, x1, y1, x2, y2 = parsed
                x1c, y1c, x2c, y2c = clip_box(x1, y1, x2, y2, W, H)

                # filter out tiny/noisy boxes
                if not passes_filters(
                    x1c, y1c, x2c, y2c, W, H,
                    min_area_rel=min_area_rel,
                    min_w_rel=min_w_rel, min_h_rel=min_h_rel,
                    min_w_px=min_w_px, min_h_px=min_h_px,
                ):
                    if verbose:
                        print(f"[skip] tiny/noisy box ({stem}:{k}) -> ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                    continue

                cx, cy, w, h = box_to_yolo_norm(x1c, y1c, x2c, y2c, W, H)
                lines_out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        else:
            if warn_on_missing_ann and verbose:
                print(f"[info] no annotation for {stem} -> writing empty label")

        if lines_out:
            n_with += 1
            n_boxes += len(lines_out)

        if not (skip_empty and not lines_out):
            write_label_file(lblp, lines_out)

        if verbose and (i % 200 == 0 or i == n_imgs):
            pass
            #print(f"[progress] {i}/{n_imgs} | with_boxes={n_with} | boxes_total={n_boxes}")

    return n_imgs, n_with, n_boxes

# --------------------------------- CLI ------------------------------------ #

def main():
    ap = argparse.ArgumentParser(description="Convert absolute bbox txts to YOLO detection labels (with tiny-box filtering).")
    ap.add_argument("--img_root", type=str, required=True, help="Root folder of images (recursively scanned).")
    ap.add_argument("--bbox_root", type=str, required=True, help="Root folder containing per-image bbox txts (same stems).")
    ap.add_argument("--out_labels", type=str, required=True, help="Output folder for YOLO detection labels.")
    ap.add_argument("--class_map", type=str, default="polyp:0", help="Mapping like 'polyp:0,adenoma:1'.")
    ap.add_argument("--default_class_id", type=int, default=0, help="Fallback class id if class name not in --class_map.")
    ap.add_argument("--quiet", action="store_true", help="Reduce logging.")
    ap.add_argument("--skip_empty", action="store_true", help="Do NOT write a file if no boxes.")

    # filtering thresholds
    ap.add_argument("--min_area_rel", type=float, default=5e-4, help="Min relative area (area/(W*H)) to keep a box. Default 0.0005 (0.05%).")
    ap.add_argument("--min_w_rel", type=float, default=0.005, help="Min relative width (w/W). Default 0.005 (0.5%).")
    ap.add_argument("--min_h_rel", type=float, default=0.005, help="Min relative height (h/H). Default 0.005 (0.5%).")
    ap.add_argument("--min_w_px", type=int, default=4, help="Min absolute width in pixels. Default 4.")
    ap.add_argument("--min_h_px", type=int, default=4, help="Min absolute height in pixels. Default 4.")

    args = ap.parse_args()

    img_root = Path(args.img_root)
    bbox_root = Path(args.bbox_root)
    out_labels = Path(args.out_labels)
    class_map = parse_class_map(args.class_map)

    n_imgs, n_with, n_boxes = process_dataset(
        img_root=img_root,
        bbox_root=bbox_root,
        out_labels=out_labels,
        class_map=class_map,
        default_class_id=args.default_class_id,
        verbose=not args.quiet,
        skip_empty=args.skip_empty,
        min_area_rel=args.min_area_rel,
        min_w_rel=args.min_w_rel,
        min_h_rel=args.min_h_rel,
        min_w_px=args.min_w_px,
        min_h_px=args.min_h_px,
    )

    print(f"[done] images={n_imgs} | with_boxes={n_with} | boxes_total={n_boxes} | labels_dir={out_labels}")

if __name__ == "__main__":
    main()
