import argparse
import json
from pathlib import Path

from PIL import Image

"""
Portable YOLO → COCO conversion script for polyp (or single-class) datasets.

- Supports two common directory layouts:
    A) images/train,val + labels/train,val
    B) train/images,labels + val/images,labels
- Detects layout automatically for each dataset root.
- Converts YOLO label files (normalized cx,cy,w,h or pixel x1,y1,x2,y2)
  into COCO-format annotations for train and val splits.
- Merges multiple dataset roots into unified COCO train/val JSONs and
  builds a roots_map.json that maps image filenames to absolute paths.
- Writes train.json, val.json, and roots_map.json into an output artifacts directory.
"""

"""
prepare_data.py

Portable YOLO → COCO converter for datasets of the form:

    (A)
    DATASET_ROOT/
      images/
        train/
        val/
      labels/
        train/
        val/

or:

    (B)
    DATASET_ROOT/
      train/
        images/
        labels/
      val/
        images/
        labels/

Run:
    python prepare_data.py \
        --datasets /path/to/D1 /path/to/D2 \
        --out-dir ./artifacts \
        --class-name polyp
"""


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def detect_dataset_layout(root: Path):
    """
    Try to detect whether dataset is in layout A (images/train ...)
    or layout B (train/images ...). Returns a dict with 4 paths.

    Raises if neither layout is found.
    """
    # layout A
    a_imgs_train = root / "images" / "train"
    a_lbls_train = root / "labels" / "train"
    a_imgs_val = root / "images" / "val"
    a_lbls_val = root / "labels" / "val"

    if a_imgs_train.exists() and a_lbls_train.exists():
        # val might not exist, that's ok, we'll still return them
        print(f"[info] detected layout A under {root}")
        return {
            "train_imgs": a_imgs_train,
            "train_lbls": a_lbls_train,
            "val_imgs": a_imgs_val,
            "val_lbls": a_lbls_val,
        }

    # layout B
    b_imgs_train = root / "train" / "images"
    b_lbls_train = root / "train" / "labels"
    b_imgs_val = root / "val" / "images"
    b_lbls_val = root / "val" / "labels"

    if b_imgs_train.exists() and b_lbls_train.exists():
        print(f"[info] detected layout B under {root}")
        return {
            "train_imgs": b_imgs_train,
            "train_lbls": b_lbls_train,
            "val_imgs": b_imgs_val,
            "val_lbls": b_lbls_val,
        }

    # if we got here → we didn't find a valid structure
    raise FileNotFoundError(
        f"[error] could not detect dataset layout under {root}.\n"
        "Expected either:\n"
        "  A) images/train + labels/train\n"
        "  B) train/images + train/labels\n"
        "Please check the dataset structure."
    )


def yolo_split_to_coco(
    img_dir: Path, lbl_dir: Path, class_name: str, start_img_id: int, start_ann_id: int
):
    """
    Convert ONE split (images/<split>, labels/<split>) to a COCO dict.
    Returns (coco_dict, next_img_id, next_ann_id).
    """
    images = []
    annotations = []
    categories = [{"id": 1, "name": class_name}]
    img_id = start_img_id
    ann_id = start_ann_id

    if not img_dir.exists():
        print(f"[warn] images dir missing: {img_dir}")
        return (
            {"images": [], "annotations": [], "categories": categories},
            img_id,
            ann_id,
        )

    # labels may be missing → that's okay, we still add images
    label_index = (
        {p.stem: p for p in lbl_dir.rglob("*.txt")} if lbl_dir.exists() else {}
    )
    if not lbl_dir.exists():
        print(
            f"[warn] labels dir missing (will create images with no boxes): {lbl_dir}"
        )

    print(f"[info] scanning images in {img_dir}")
    img_files = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"[info] found {len(img_files)} images in {img_dir}")

    matched = parsed = skipped = 0

    for img_path in sorted(img_files):
        # open image for size
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception as e:
            print(f"[warn] cannot open image {img_path}: {e}")
            continue

        images.append(
            {
                "id": img_id,
                # store absolute path so training is easy later
                "file_name": str(img_path.resolve()),
                "width": W,
                "height": H,
            }
        )

        # try to find label
        lf = label_index.get(img_path.stem)
        if lf and lf.exists():
            matched += 1
            for line in lf.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                ok = False
                # try YOLO normalized: cls cx cy w h
                try:
                    cx, cy, w, h = map(float, parts[1:5])
                    if 0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1:
                        x = (cx - w / 2) * W
                        y = (cy - h / 2) * H
                        bw = w * W
                        bh = h * H
                        ok = True
                except Exception:
                    ok = False

                # fallback: pixels x1 y1 x2 y2
                if not ok:
                    try:
                        x1, y1, x2, y2 = map(float, parts[1:5])
                        if x2 > x1 and y2 > y1:
                            x = max(0.0, min(x1, W - 1))
                            y = max(0.0, min(y1, H - 1))
                            bw = max(0.0, min(x2, W - 1) - x)
                            bh = max(0.0, min(y2, H - 1) - y)
                            ok = bw > 1 and bh > 1
                    except Exception:
                        ok = False

                if not ok:
                    skipped += 1
                    continue

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [float(x), float(y), float(bw), float(bh)],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                    }
                )
                ann_id += 1
                parsed += 1

        img_id += 1

    print(
        f"[diag] {img_dir}: images={len(images)} matched_labels={matched} boxes={parsed} skipped={skipped}"
    )

    coco = {"images": images, "annotations": annotations, "categories": categories}
    return coco, img_id, ann_id


def merge_cocos(cocos):
    """Merge several COCO dicts into a single one, remapping IDs."""
    if not cocos:
        return {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "polyp"}],
        }

    merged = {"images": [], "annotations": [], "categories": cocos[0]["categories"]}
    new_img_id = 1
    new_ann_id = 1

    for coco in cocos:
        oldid_to_newid = {}
        for img in coco["images"]:
            old_id = img["id"]
            oldid_to_newid[old_id] = new_img_id
            merged["images"].append({**img, "id": new_img_id})
            new_img_id += 1
        for ann in coco["annotations"]:
            merged["annotations"].append(
                {
                    **ann,
                    "id": new_ann_id,
                    "image_id": oldid_to_newid[ann["image_id"]],
                }
            )
            new_ann_id += 1

    return merged


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)
    print(f"[info] wrote {path}")


def main():
    parser = argparse.ArgumentParser("Prepare COCO from YOLO datasets (portable)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="1+ dataset roots, each with EITHER images/train,val + labels/train,val OR train/images,labels + val/images,labels",
    )
    parser.add_argument("--out-dir", default="./artifacts")
    parser.add_argument("--class-name", default="polyp")
    parser.add_argument("--train-json", default="train.json")
    parser.add_argument("--val-json", default="val.json")
    parser.add_argument("--roots-map", default="roots_map.json")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_train = []
    all_val = []
    roots_map = {}

    next_img_id = 1
    next_ann_id = 1

    for ds in args.datasets:
        ds = Path(ds)
        print(f"[info] processing dataset: {ds.resolve()}")

        # 🔥 NEW: detect layout here
        layout = detect_dataset_layout(ds)
        imgs_train = layout["train_imgs"]
        lbls_train = layout["train_lbls"]
        imgs_val = layout["val_imgs"]
        lbls_val = layout["val_lbls"]

        print(f"   train images: {imgs_train.exists()} → {imgs_train}")
        print(f"   train labels: {lbls_train.exists()} → {lbls_train}")
        print(f"   val images:   {imgs_val.exists()} → {imgs_val}")
        print(f"   val labels:   {lbls_val.exists()} → {lbls_val}")

        # TRAIN
        coco_tr, next_img_id, next_ann_id = yolo_split_to_coco(
            imgs_train,
            lbls_train,
            class_name=args.class_name,
            start_img_id=next_img_id,
            start_ann_id=next_ann_id,
        )
        if coco_tr["images"]:
            all_train.append(coco_tr)
            for img in imgs_train.rglob("*"):
                if img.suffix.lower() in IMG_EXTS:
                    roots_map[img.name] = str(img.resolve())

        # VAL
        coco_va, next_img_id, next_ann_id = yolo_split_to_coco(
            imgs_val,
            lbls_val,
            class_name=args.class_name,
            start_img_id=next_img_id,
            start_ann_id=next_ann_id,
        )
        if coco_va["images"]:
            all_val.append(coco_va)
            for img in imgs_val.rglob("*"):
                if img.suffix.lower() in IMG_EXTS:
                    roots_map[img.name] = str(img.resolve())

    merged_train = merge_cocos(all_train)
    merged_val = merge_cocos(all_val)

    save_json(merged_train, out_dir / args.train_json)
    save_json(merged_val, out_dir / args.val_json)
    save_json(roots_map, out_dir / args.roots_map)


if __name__ == "__main__":
    main()
