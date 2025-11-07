#!/usr/bin/env python3
"""
build_seq_yolo_det_labels.py  (EXACT folder matching)
-----------------------------------------------------
Positive sequences:
  seq3/ or seq_4/  → must contain BOTH:
    images_seq3  or images_seq_3
    bbox_seq3    or bbox_seq_3
Negative sequences:
  seq3_neg/ or seq_4_neg/  → images only; randomly select a fraction to emit EMPTY labels.

Exact matching prevents accidentally picking similarly named folders.

Logging:
  - Only prints [skip] for tiny/noisy boxes and a final [done] line by default.
  - Use --verbose 1 for per-sequence summaries.
"""

from __future__ import annotations
from pathlib import Path
import argparse
import random
import re
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# -------------------------- tiny/noise filters --------------------------- #

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
    area_rel = (w_px * h_px) / float(max(W * H, 1))
    if area_rel < min_area_rel:
        return False
    if (w_px / W) < min_w_rel or (h_px / H) < min_h_rel:
        return False
    if w_px < min_w_px or h_px < min_h_px:
        return False
    return True

# ---------------------------- IO utilities ------------------------------ #

def write_label(lbl_path: Path, lines: List[str]) -> None:
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.write_text("\n".join(lines))

def list_images(d: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]

# ---------------------- exact name / sequence parsing -------------------- #

_POS_SEQ_RE = re.compile(r"^seq_?(\d+)$", re.IGNORECASE)
_NEG_SEQ_RE = re.compile(r"^seq_?(\d+)_neg$", re.IGNORECASE)

def parse_seq_index(name: str) -> Optional[int]:
    """Return sequence index for positive seq folder names like 'seq3' or 'seq_4'."""
    m = _POS_SEQ_RE.match(name)
    if m:
        return int(m.group(1))
    return None

def is_negative_seq(name: str) -> Optional[int]:
    """Return sequence index for negative seq folder names like 'seq3_neg' or 'seq_4_neg'."""
    m = _NEG_SEQ_RE.match(name)
    if m:
        return int(m.group(1))
    return None

def find_exact_subdir(seq_dir: Path, expected_names: List[str]) -> Optional[Path]:
    """
    Return the direct child whose name equals one of expected_names (case-insensitive).
    No 'contains' or fuzzy match—exact equality only (ignoring case).
    """
    expected_lower = {n.lower() for n in expected_names}
    for d in sorted(seq_dir.iterdir()):
        if d.is_dir() and d.name.lower() in expected_lower:
            return d
    return None

# ------------------------- bbox parsing / math --------------------------- #

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

def parse_bbox_line(line: str, class_map: Dict[str,int], default_class_id: int) -> Optional[Tuple[int,float,float,float,float]]:
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

def clip_box(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float,float,float,float]:
    x1c = min(max(x1, 0.0), max(W - 1.0, 0.0))
    y1c = min(max(y1, 0.0), max(H - 1.0, 0.0))
    x2c = min(max(x2, 0.0), max(W - 1.0, 0.0))
    y2c = min(max(y2, 0.0), max(H - 1.0, 0.0))
    xl, xr = min(x1c, x2c), max(x1c, x2c)
    yt, yb = min(y1c, y2c), max(y1c, y2c)
    return xl, yt, xr, yb

def box_to_yolo_norm(x1: float, y1: float, x2: float, y2: float, W: int, H: int) -> Tuple[float,float,float,float]:
    w = max(x2 - x1, 0.0)
    h = max(y2 - y1, 0.0)
    cx = x1 + w / 2.0
    cy = y1 + h / 2.0
    return cx / W, cy / H, w / W, h / H

# ---------------------------- Core routines ------------------------------ #

def process_positive_sequence(
    seq_dir: Path,
    out_root: Path,
    mirror: bool,
    seq_idx: int,
    class_map: Dict[str,int],
    default_class_id: int,
    min_area_rel: float,
    min_w_rel: float, min_h_rel: float,
    min_w_px: int, min_h_px: int,
    verbose: int,
) -> Tuple[int,int,int]:
    """
    Positive sequence: require exact subdirs:
      images_seq{idx} or images_seq_{idx}
      bbox_seq{idx}   or bbox_seq_{idx}
    """
    img_candidates = [f"images_seq{seq_idx}", f"images_seq_{seq_idx}"]
    bbox_candidates = [f"bbox_seq{seq_idx}", f"bbox_seq_{seq_idx}"]

    images_dir = find_exact_subdir(seq_dir, img_candidates)
    bbox_dir   = find_exact_subdir(seq_dir, bbox_candidates)

    if images_dir is None or bbox_dir is None:
        if verbose:
            print(f"[warn] {seq_dir.name}: missing exact 'images_seq{seq_idx}'/'bbox_seq{seq_idx}' (or underscored) → skipped")
        return 0, 0, 0

    out_dir = (out_root / seq_dir.name) if mirror else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = list_images(images_dir, recursive=False)
    n_imgs = len(imgs)
    n_with = 0
    n_boxes_total = 0

    for ip in imgs:
        im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if im is None:
            if verbose:
                print(f"[warn] failed to read image: {ip}")
            continue
        H, W = im.shape[:2]
        stem = ip.stem

        annp = bbox_dir / f"{stem}.txt"
        lines_out: List[str] = []

        if annp.exists():
            raw = annp.read_text(encoding="utf-8", errors="ignore").splitlines()
            for k, line in enumerate(raw, 1):
                parsed = parse_bbox_line(line, class_map, default_class_id)
                if parsed is None:
                    continue
                cls_id, x1, y1, x2, y2 = parsed
                x1c, y1c, x2c, y2c = clip_box(x1, y1, x2, y2, W, H)
                if not passes_filters(
                    x1c, y1c, x2c, y2c, W, H,
                    min_area_rel=min_area_rel,
                    min_w_rel=min_w_rel, min_h_rel=min_h_rel,
                    min_w_px=min_w_px, min_h_px=min_h_px
                ):
                    print(f"[skip] tiny/noisy box ({seq_dir.name}/{stem}:{k}) -> ({x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f})")
                    continue
                cx, cy, w, h = box_to_yolo_norm(x1c, y1c, x2c, y2c, W, H)
                lines_out.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

        if lines_out:
            n_with += 1
            n_boxes_total += len(lines_out)

        write_label(out_dir / f"{stem}.txt", lines_out)

    if verbose:
        print(f"[pos] {seq_dir.name}: images={n_imgs} | with_boxes={n_with} | boxes_total={n_boxes_total}")
    return n_imgs, n_with, n_boxes_total


def process_negative_sequence(
    seq_dir: Path,
    out_root: Path,
    mirror: bool,
    recursive_neg: bool,
    fraction: float,
    rng: random.Random,
    verbose: int,
) -> int:
    """
    Negative sequence: randomly select a fraction of images and write EMPTY labels.
    Returns: number of empty labels written.
    """
    imgs = list_images(seq_dir, recursive=recursive_neg)
    if not imgs:
        if verbose:
            print(f"[neg] {seq_dir.name}: no images found")
        return 0

    k = max(1, int(round(fraction * len(imgs)))) if fraction > 0.0 else 0
    picked = rng.sample(imgs, k) if k > 0 else []

    out_dir = (out_root / seq_dir.name) if mirror else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    for ip in picked:
        write_label(out_dir / f"{ip.stem}.txt", [])  # empty label

    if verbose:
        print(f"[neg] {seq_dir.name}: selected={len(picked)}/{len(imgs)} (fraction={fraction:.3f})")
    return len(picked)


def main():
    ap = argparse.ArgumentParser(description="Build YOLO detection labels for positive/negative sequences (exact subfolder names).")
    ap.add_argument("--root", type=str, required=True,
                    help="Root containing sequences: seq3/, seq_4/, seq3_neg/, seq_4_neg/, ...")
    ap.add_argument("--out_labels", type=str, required=True,
                    help="Root directory to write YOLO det .txt labels")
    ap.add_argument("--mirror", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=True,
                    help="Mirror per-seq subfolders under out_labels (default True)")
    ap.add_argument("--recursive_neg", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=False,
                    help="Recursively scan negative seq folders (default False)")
    ap.add_argument("--neg_fraction", type=float, default=1.0/3.0,
                    help="Fraction of images to select per negative sequence (default 0.3333)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for negative selection")

    # class + filters
    ap.add_argument("--class_map", type=str, default="polyp:0",
                    help="Mapping like 'polyp:0,adenoma:1'. Default 'polyp:0'")
    ap.add_argument("--default_class_id", type=int, default=0,
                    help="Fallback class id if name not in --class_map")
    ap.add_argument("--min_area_rel", type=float, default=0.001,
                    help="Min relative area (area/(W*H)) for boxes (default 0.001)")
    ap.add_argument("--min_w_rel", type=float, default=0.004,
                    help="Min relative width (w/W) (default 0.004)")
    ap.add_argument("--min_h_rel", type=float, default=0.004,
                    help="Min relative height (h/H) (default 0.004)")
    ap.add_argument("--min_w_px", type=int, default=4,
                    help="Min absolute width in pixels (default 4)")
    ap.add_argument("--min_h_px", type=int, default=4,
                    help="Min absolute height in pixels (default 4)")

    # logging
    ap.add_argument("--verbose", type=int, default=0,
                    help="0 = only [skip] & final [done]; 1 = also per-sequence summaries")

    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_labels)
    out_root.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    class_map = parse_class_map(args.class_map)

    total_pos_imgs = total_pos_with = total_pos_boxes = 0
    total_neg_selected = 0

    seq_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    for sd in seq_dirs:
        name = sd.name
        n_idx = is_negative_seq(name)
        if n_idx is not None:
            total_neg_selected += process_negative_sequence(
                sd, out_root, args.mirror, args.recursive_neg,
                fraction=args.neg_fraction, rng=rng, verbose=args.verbose
            )
            continue

        p_idx = parse_seq_index(name)
        if p_idx is not None:
            n_imgs, n_with, n_boxes = process_positive_sequence(
                sd, out_root, args.mirror,
                seq_idx=p_idx,
                class_map=class_map, default_class_id=args.default_class_id,
                min_area_rel=args.min_area_rel,
                min_w_rel=args.min_w_rel, min_h_rel=args.min_h_rel,
                min_w_px=args.min_w_px, min_h_px=args.min_h_px,
                verbose=args.verbose
            )
            total_pos_imgs += n_imgs
            total_pos_with += n_with
            total_pos_boxes += n_boxes
            continue

        if args.verbose:
            print(f"[skip] {sd.name}: not a recognized seq folder")

    # Final summary only (minimal logging)
    print(f"[done] pos_images={total_pos_imgs} | pos_with_boxes={total_pos_with} | pos_boxes_total={total_pos_boxes} | "
          f"neg_empty_selected={total_neg_selected} | labels_root={out_root}")

if __name__ == "__main__":
    main()
