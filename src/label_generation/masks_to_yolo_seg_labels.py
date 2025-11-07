"""
masks_to_yolo_seg_labels.py
---------------------------
Generate YOLO segmentation label .txt files from binary mask images.

For each image in --img_root, this script searches across one or more --mask_roots
for a corresponding mask file named '<image_stem>_mask.(png|jpg|jpeg|bmp|tif|tiff)'.
It extracts external contours as polygons, simplifies them, normalizes coordinates,
and writes YOLOv8-seg style labels:

  class_id x1 y1 x2 y2 x3 y3 ...   (one line per instance)

Defaults assume a single class = 0 (polyp). Multiple instances are supported.

Usage (example from Jupyter or shell):
!python ../src/masks_to_yolo_seg_labels.py \
  --img_root "../data/detection2/train/images_single" \
  --mask_roots "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C1" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C2" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C3" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C4" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C5" \
               "../data/PolypGen2021_MultiCenterData_v3/data_C1/masks_C6" \
  --out_labels "../data/detection2/train/labels_single_seg" \
  --approx_eps 1.5 \
  --min_area_px 25
"""

from pathlib import Path
import argparse
import cv2
import numpy as np
from typing import List, Optional, Tuple

IMG_EXTS  = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = IMG_EXTS  # allow same set

def find_mask_for_image(stem: str, mask_roots: List[Path]) -> Optional[Path]:
    """Search all mask_roots for a file named '<stem>_mask.*' with known extensions."""
    candidates = []
    for root in mask_roots:
        for ext in MASK_EXTS:
            p = root / f"{stem}_mask{ext}"
            if p.exists():
                candidates.append(p)
            #Try <stem>.ext directly (no '_mask' suffix)
            p2 = root / f"{stem}{ext}"
            if p2.exists():
                candidates.append(p2)
    if not candidates:
        return None
    # Prefer PNG if multiple found, else first by name sort.
    candidates = sorted(candidates, key=lambda p: (p.suffix.lower() != ".png", str(p)))
    return candidates[0]

def mask_to_polygons(mask: np.ndarray,
                     approx_eps: float = 1.5,
                     min_area_px: int = 25) -> List[np.ndarray]:
    """
    Convert a binary mask (H,W) to simplified external polygons.
    - approx_eps: epsilon for Douglas-Peucker (in pixels).
    - min_area_px: discard very small components by area.
    Returns list of polygons as (N,2) float arrays in (x,y) image coords.
    """
    # Ensure binary uint8 {0,255}
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    _, binm = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)
    # Find external contours only (holes ignored)
    cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < float(min_area_px):
            continue
        if approx_eps > 0:
            c = cv2.approxPolyDP(c, epsilon=approx_eps, closed=True)
        c = c.reshape(-1, 2).astype(np.float32)
        # Need at least 3 points
        if c.shape[0] >= 3:
            polys.append(c)
    return polys

def normalize_polygon(poly: np.ndarray, W: int, H: int) -> np.ndarray:
    """Normalize polygon points to [0,1] by (x/W, y/H), and clip to [0,1]."""
    out = poly.copy().astype(np.float32)
    out[:, 0] = np.clip(out[:, 0] / float(W), 0.0, 1.0)
    out[:, 1] = np.clip(out[:, 1] / float(H), 0.0, 1.0)
    return out

def polygon_to_yolo_line(poly_norm: np.ndarray, class_id: int = 0) -> str:
    """
    Convert normalized polygon (N,2) to YOLO seg label line:
    'class_id x1 y1 x2 y2 ...'
    """
    coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in poly_norm])
    return f"{class_id} {coords}"

def write_label_file(lbl_path: Path, lines: List[str]) -> None:
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    if lines:
        lbl_path.write_text("\n".join(lines))
    else:
        # Empty file is acceptable (no instances); or you can skip writing.
        lbl_path.write_text("")

def process_dataset(img_root: Path,
                    mask_roots: List[Path],
                    out_labels: Path,
                    class_id: int = 0,
                    approx_eps: float = 1.5,
                    min_area_px: int = 25,
                    verbose: bool = True) -> Tuple[int,int,int]:
    """
    Iterate images, locate masks, extract polygons, and write YOLO seg labels.
    Returns (num_images, num_with_masks, num_instances_total).
    """
    img_files = [p for p in sorted(img_root.iterdir()) if p.suffix.lower() in IMG_EXTS]
    n_imgs = len(img_files)
    n_with_mask = 0
    n_inst = 0

    for i, ip in enumerate(img_files, 1):
        im = cv2.imread(str(ip), cv2.IMREAD_UNCHANGED)
        if im is None:
            if verbose:
                print(f"[warn] failed to read image: {ip}")
            continue
        H, W = im.shape[:2]
        stem = ip.stem

        mp = find_mask_for_image(stem, mask_roots)
        if mp is None:
            # no mask found; write empty label (or skip)
            lblp = out_labels / f"{stem}.txt"
            write_label_file(lblp, [])
            if verbose:
                print(f"[info] no mask for {stem}, wrote empty label")
            continue

        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            if verbose:
                print(f"[warn] failed to read mask: {mp}")
            lblp = out_labels / f"{stem}.txt"
            write_label_file(lblp, [])
            continue

        polys = mask_to_polygons(mask, approx_eps=approx_eps, min_area_px=min_area_px)
        lines = []
        for poly in polys:
            poly_norm = normalize_polygon(poly, W=W, H=H)
            # (optional) limit polygon length to YOLOv8's practical limits
            if poly_norm.shape[0] > 1000:
                # uniform subsample if extremely dense
                idx = np.linspace(0, poly_norm.shape[0]-1, 1000).astype(int)
                poly_norm = poly_norm[idx]
            line = polygon_to_yolo_line(poly_norm, class_id=class_id)
            lines.append(line)

        lblp = out_labels / f"{stem}.txt"
        write_label_file(lblp, lines)

        if lines:
            n_with_mask += 1
            n_inst += len(lines)

        if verbose and (i % 200 == 0 or i == n_imgs):
            print(f"[progress] {i}/{n_imgs} | with_mask={n_with_mask} | instances={n_inst}")

    return n_imgs, n_with_mask, n_inst

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img_root", type=str, required=True,
                    help="Directory containing images")
    ap.add_argument("--mask_roots", type=str, nargs="+", required=True,
                    help="One or more directories that contain '<stem>_mask.*' files")
    ap.add_argument("--out_labels", type=str, required=True,
                    help="Output directory for YOLO-seg .txt labels")
    ap.add_argument("--class_id", type=int, default=0,
                    help="Class id to write (default 0)")
    ap.add_argument("--approx_eps", type=float, default=1.5,
                    help="Douglas-Peucker epsilon in pixels to simplify polygons")
    ap.add_argument("--min_area_px", type=int, default=25,
                    help="Minimum connected component area in pixels to keep")
    ap.add_argument("--quiet", action="store_true",
                    help="Reduce logging")
    args = ap.parse_args()

    img_root = Path(args.img_root)
    mask_roots = [Path(p) for p in args.mask_roots]
    out_labels = Path(args.out_labels)

    n_imgs, n_with_mask, n_inst = process_dataset(
        img_root=img_root,
        mask_roots=mask_roots,
        out_labels=out_labels,
        class_id=args.class_id,
        approx_eps=args.approx_eps,
        min_area_px=args.min_area_px,
        verbose=not args.quiet
    )

    print(f"[done] images={n_imgs} | with_mask={n_with_mask} | instances={n_inst} | labels_dir={out_labels}")

if __name__ == "__main__":
    main()
