"""
combine_images_and_labels.py
----------------------------
Flatten and merge images (and their YOLO-seg labels) from multiple sources into one folder.

Rules:
- Only include images that have a matching label file.
- Never create empty labels.
- Handles nested label folders automatically.
- Avoids duplicate images (by md5 hash).

Usage example:
!python ../src/combine_images_and_labels.py \
  --train_images_root "../data/detection2/train" \
  --extra_images "../data/segmentation2/aug_neg_images" "../data/segmentation2/aug_pos_images" \
  --labels_root "../data/segmentation2/yolo_split2/train_yolo_labels" \
  --output_root "../data/segmentation2/yolo_split2/train"
"""

import argparse
from pathlib import Path
import shutil
import hashlib


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_seq_images(seq_root: Path):
    if not seq_root.exists():
        return
    for seq_dir in sorted(seq_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        name = seq_dir.name.lower()
        if name.endswith("_neg") or "_neg" in name:
            for p in sorted(seq_dir.iterdir()):
                if is_image(p):
                    yield p
        else:
            for sub in sorted(seq_dir.iterdir()):
                if sub.is_dir() and "images" in sub.name.lower():
                    for p in sorted(sub.iterdir()):
                        if is_image(p):
                            yield p


def collect_all_images(train_images_root: Path, extra_images: list[Path]):
    sources = []
    single = train_images_root / "images_single"
    if single.exists():
        sources.extend([p for p in single.iterdir() if is_image(p)])
    seq_root = train_images_root / "seq"
    sources.extend(list(collect_seq_images(seq_root)))
    for extra in extra_images:
        if extra.exists():
            sources.extend([p for p in extra.iterdir() if is_image(p)])
    return sources


def build_label_index(labels_root: Path):
    index = {}
    for p in labels_root.rglob("*.txt"):
        stem = p.stem
        info = (p.stat().st_size > 0, p.stat().st_mtime)
        if stem not in index or info > (index[stem][0], index[stem][1]):
            index[stem] = (info[0], info[1], p)
    return {k: v[2] for k, v in index.items()}


def combine(train_images_root: Path, extra_images: list[Path],
            labels_root: Path, output_root: Path):

    target_images = output_root / "images"
    target_labels = output_root / "labels"
    target_images.mkdir(parents=True, exist_ok=True)
    target_labels.mkdir(parents=True, exist_ok=True)

    lbl_index = build_label_index(labels_root)
    img_paths = collect_all_images(train_images_root, extra_images)
    print(f"[info] found {len(img_paths)} total images before filtering")

    seen_hash = {}
    stem_counts = {}
    copied, labeled, skipped_dupes, skipped_no_label = 0, 0, 0, 0

    for src in img_paths:
        stem = src.stem
        lbl_src = lbl_index.get(stem)
        if not lbl_src or not lbl_src.exists():
            skipped_no_label += 1
            continue  # strict match only

        sig = md5sum(src)
        if sig in seen_hash:
            skipped_dupes += 1
            continue
        seen_hash[sig] = src

        ext = src.suffix.lower()
        out_stem = stem
        if out_stem in stem_counts:
            stem_counts[out_stem] += 1
            out_stem = f"{stem}_{stem_counts[out_stem]:03d}"
        else:
            stem_counts[out_stem] = 0

        # copy image + label pair
        dst_img = target_images / f"{out_stem}{ext}"
        dst_lbl = target_labels / f"{out_stem}.txt"
        shutil.copy2(src, dst_img)
        shutil.copy2(lbl_src, dst_lbl)
        copied += 1
        labeled += 1

    print("\n[done]")
    print(f"  image-label pairs copied : {copied}")
    print(f"  images skipped (no label): {skipped_no_label}")
    print(f"  duplicates skipped       : {skipped_dupes}")
    print(f"\nTargets:")
    print(f"  images -> {target_images}")
    print(f"  labels -> {target_labels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_images_root", type=str, required=True,
                    help="Root folder containing images_single and seq subfolders")
    ap.add_argument("--extra_images", type=str, nargs="*", default=[],
                    help="Optional extra image folders (e.g., augmented images)")
    ap.add_argument("--labels_root", type=str, required=True,
                    help="Folder containing YOLO labels (can include subfolders)")
    ap.add_argument("--output_root", type=str, required=True,
                    help="Output folder (will create images/ and labels/ inside)")
    args = ap.parse_args()

    combine(Path(args.train_images_root),
            [Path(p) for p in args.extra_images],
            Path(args.labels_root),
            Path(args.output_root))


if __name__ == "__main__":
    main()
