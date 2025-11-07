"""
build_seq_yolo_seg_labels.py
----------------------------
Process a dataset root containing many sequence folders:
  - Positive sequences:  seq1/, seq2/, ... each contains subfolders like:
        images_seq1/ , mask_seq1/   (names may vary: any 'images*' & 'mask*')
    → For every image in images_*, find the corresponding mask in mask_*,
      extract polygons, and write YOLOv8-seg labels (one line per instance).

  - Negative sequences:  seq3_neg/, seq10_neg/, ...
    → For every image in the folder (or its subfolders), write an EMPTY label.

Output labels can be mirrored by sequence into the out root:
  out_labels/
    seq1/
      img001.txt
    seq2/
      ...
    seq3_neg/
      ...
or flattened (all labels directly under out_labels/).

USAGE
-----
!python ../src/build_seq_yolo_seg_labels.py \
  --root "../data/segmentation2/sequences" \
  --out_labels "../data/segmentation2/yolo_seg_labels" \
  --mirror True \
  --approx_eps 1.5 \
  --min_area_px 25

OPTIONS
-------
--root          : dataset root containing many seq folders (seq*, seq*_neg)
--out_labels    : destination root for .txt labels
--mirror        : if True, keep per-sequence subfolders under out_labels (default True)
--recursive_neg : if True, scan negative seq folders recursively for images (default False)
--class_id      : numeric class id to write (default 0)
--approx_eps    : Douglas-Peucker epsilon (pixels) for polygon simplification
--min_area_px   : minimum connected-component area (pixels) to keep
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
import re
from typing import Dict, List, Optional, Tuple

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMG_EXTS

# --- Helpers for robust name matching ---
STRIP_SUFFIXES = [
    "_mask", "-mask", ".mask",
    "_msk",  "-msk",
    "_gt",   "-gt", "_gt1", "_gt2",
    "_seg",  "-seg",
    "_annotation", "-annotation",
]

def norm_stem(name: str) -> str:
    """Lowercase, strip extension, then common mask suffixes."""
    stem = name.lower()
    stem = re.sub(r"\.[a-z0-9]+$", "", stem)  # drop extension
    changed = True
    while changed:
        changed = False
        for suf in STRIP_SUFFIXES:
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                changed = True
    return stem

# --- Polygon / label utils ---
def mask_to_polygons(mask: np.ndarray, approx_eps: float, min_area_px: int) -> List[np.ndarray]:
    if mask.dtype != np.uint8:
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

def normalize_polygon(poly_xy: np.ndarray, W: int, H: int) -> np.ndarray:
    out = poly_xy.copy().astype(np.float32)
    out[:, 0] = np.clip(out[:, 0] / float(W), 0.0, 1.0)  # x/W
    out[:, 1] = np.clip(out[:, 1] / float(H), 0.0, 1.0)  # y/H
    return out

def poly_to_yolo_line(poly_norm: np.ndarray, class_id: int = 0) -> str:
    coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly_norm])
    return f"{class_id} {coords}"

def write_label(lbl_path: Path, lines: List[str]) -> None:
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        lbl_path.write_text("\n".join(lines))
    else:
        lbl_path.write_text("")  # empty = no instances / negative

# --- Core processing ---
def find_subdir(seq_dir: Path, key: str) -> Optional[Path]:
    """Return the first subfolder whose name contains 'key' (case-insensitive)."""
    for d in sorted(seq_dir.iterdir()):
        if d.is_dir() and key in d.name.lower():
            return d
    return None

def build_mask_index(mask_dir: Path) -> Dict[str, Path]:
    """
    Index mask files (non-recursive) by normalized stem.
    If you need recursive for masks, switch to rglob here.
    """
    idx: Dict[str, Path] = {}
    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            ns = norm_stem(p.stem)
            if ns not in idx:
                idx[ns] = p
    return idx

def process_positive_sequence(
    seq_dir: Path,
    out_root: Path,
    mirror: bool,
    class_id: int,
    approx_eps: float,
    min_area_px: int,
) -> Tuple[int, int]:
    """
    Process one positive sequence folder:
      - find images_* and mask_* subdirs
      - for each image, match mask, write YOLO-seg labels
    Returns: (n_images_seen, n_images_with_any_instance)
    """
    images_dir = find_subdir(seq_dir, "images")
    mask_dir   = find_subdir(seq_dir, "mask")

    if images_dir is None or mask_dir is None:
        print(f"[warn] {seq_dir.name}: missing 'images_*' or 'mask_*' folder → skipped")
        return 0, 0

    # label destination (mirror or flat)
    out_dir = (out_root / seq_dir.name) if mirror else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    # index masks by normalized stem
    mindex = build_mask_index(mask_dir)

    imgs = [p for p in sorted(images_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMG_EXTS]

    n = len(imgs)
    n_with = 0

    for ip in imgs:
        im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if im is None:
            print(f"[warn] failed to read image: {ip}")
            continue
        H, W = im.shape[:2]

        stem = ip.stem
        # try exact stem first; if missing, try normalized
        mp = mindex.get(stem)
        if mp is None:
            ns = norm_stem(stem)
            mp = mindex.get(ns)

        lines: List[str] = []
        if mp is not None and mp.exists():
            mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"[warn] cannot read mask: {mp}")
            else:
                polys = mask_to_polygons(mask, approx_eps=approx_eps, min_area_px=min_area_px)
                for poly in polys:
                    poly_n = normalize_polygon(poly, W, H)
                    # (optional cap) if polygon is extremely dense:
                    if poly_n.shape[0] > 1000:
                        idx = np.linspace(0, poly_n.shape[0]-1, 1000).astype(int)
                        poly_n = poly_n[idx]
                    lines.append(poly_to_yolo_line(poly_n, class_id=class_id))

        if lines:
            n_with += 1

        lblp = out_dir / f"{stem}.txt"
        write_label(lblp, lines)

    return n, n_with

def iter_images_in_dir(d: Path, recursive: bool = False) -> List[Path]:
    if recursive:
        return [p for p in sorted(d.rglob("*")) if p.is_file() and p.suffix.lower() in IMG_EXTS]
    else:
        return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]

def process_negative_sequence(
    seq_dir: Path,
    out_root: Path,
    mirror: bool,
    recursive_neg: bool
) -> int:
    """
    Process one negative sequence folder:
      - read images under seq_dir (or recursively)
      - write EMPTY label files
    Returns: number of images processed.
    """
    out_dir = (out_root / seq_dir.name) if mirror else out_root
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = iter_images_in_dir(seq_dir, recursive=recursive_neg)
    for ip in imgs:
        lblp = out_dir / f"{ip.stem}.txt"
        write_label(lblp, [])
    return len(imgs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="Root containing many sequences: seq1/, seq2/, seq3_neg/, ...")
    ap.add_argument("--out_labels", type=str, required=True,
                    help="Root directory to write YOLO-seg .txt labels")
    ap.add_argument("--mirror", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=True,
                    help="Mirror per-seq subfolders under out_labels (default True)")
    ap.add_argument("--recursive_neg", type=lambda x: str(x).lower() in {"1","true","yes","y"}, default=False,
                    help="If True, scan negative seq folders recursively for images (default False)")
    ap.add_argument("--class_id", type=int, default=0)
    ap.add_argument("--approx_eps", type=float, default=1.5)
    ap.add_argument("--min_area_px", type=int, default=25)
    args = ap.parse_args()

    root = Path(args.root)
    out_root = Path(args.out_labels)
    out_root.mkdir(parents=True, exist_ok=True)

    seq_dirs = [d for d in sorted(root.iterdir()) if d.is_dir()]
    total_pos_imgs = total_pos_with = total_neg_imgs = 0

    for sd in seq_dirs:
        name = sd.name.lower()
        if name.endswith("_neg") or "_neg" in name:
            nneg = process_negative_sequence(sd, out_root, args.mirror, args.recursive_neg)
            total_neg_imgs += nneg
            print(f"[neg]  {sd.name}: wrote {nneg} empty labels")
        elif name.startswith("seq"):
            n, w = process_positive_sequence(
                sd, out_root, args.mirror, args.class_id, args.approx_eps, args.min_area_px
            )
            total_pos_imgs += n
            total_pos_with += w
            print(f"[pos]  {sd.name}: images={n} | with_instances={w}")
        else:
            # Skip non-seq folders
            print(f"[skip] {sd.name}: not a seq folder")

    print("\n[done]")
    print(f"  positive images: {total_pos_imgs} | with instances: {total_pos_with}")
    print(f"  negative images (empty labels): {total_neg_imgs}")
    print(f"  labels root: {out_root}")

if __name__ == "__main__":
    main()
