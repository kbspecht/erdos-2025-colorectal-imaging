import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms import functional as TF

"""
Run Faster R-CNN inference on a folder of images and visualize GT vs predictions.

- Parses CLI args for model checkpoint, images directory, output directory,
  optional COCO-style JSON (for ground-truth boxes), thresholds, and device.
- Loads a Faster R-CNN model from weights and runs inference on each image.
- Optionally looks up COCO ground-truth boxes and overlays them in green.
- Applies score filtering and NMS, then overlays predicted boxes in red
  with confidence scores.
- Saves side-by-side visualizations (original | GT+predictions) to the
  specified output directory.
"""


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))


# --- make src/ the package root so "faster_rcnn" is importable ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from faster_rcnn.model_builder_faster_rcnn import build_fasterrcnn  # noqa: E402

# ----------------- helpers -----------------


def load_coco_annotations(coco_json: Path):
    """
    Load COCO-style annotations and build:
      - full_name_to_id:  image["file_name"] -> image_id
      - base_name_to_id:  Path(image["file_name"]).name -> image_id
      - annos_by_imgid:   image_id -> list of annotations
      - catid_to_name:    category_id -> name
    """
    with coco_json.open("r") as f:
        data = json.load(f)

    full_name_to_id = {}
    base_name_to_id = {}
    for img in data.get("images", []):
        img_id = img["id"]
        fname = img["file_name"]
        full_name_to_id[fname] = img_id
        base_name_to_id[Path(fname).name] = img_id

    annos_by_imgid = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        annos_by_imgid.setdefault(img_id, []).append(ann)

    catid_to_name = {}
    for cat in data.get("categories", []):
        catid_to_name[cat["id"]] = cat.get("name", f"class_{cat['id']}")

    return full_name_to_id, base_name_to_id, annos_by_imgid, catid_to_name


def get_gt_for_image(
    img_path: Path,
    full_name_to_id,
    base_name_to_id,
    annos_by_imgid,
):
    """
    Return list of GT annotations for a given image file.
    Tries:
      1) basename match,
      2) any COCO file_name whose basename matches,
      3) suffix match (slow but robust for nested paths).
    """
    fname = img_path.name

    img_id = None
    # 1) direct basename mapping
    if fname in base_name_to_id:
        img_id = base_name_to_id[fname]
    else:
        # 2) try any full name whose basename matches
        for full_name, _id in full_name_to_id.items():
            if Path(full_name).name == fname:
                img_id = _id
                break
        # 3) last resort: suffix match
        if img_id is None:
            for full_name, _id in full_name_to_id.items():
                if full_name.endswith(fname):
                    img_id = _id
                    break

    if img_id is None:
        return []

    return annos_by_imgid.get(img_id, [])


def make_side_by_side(left: Image.Image, right: Image.Image) -> Image.Image:
    w, h = left.size
    canvas = Image.new("RGB", (2 * w, h), (0, 0, 0))
    canvas.paste(left, (0, 0))
    canvas.paste(right, (w, 0))
    return canvas


# ----------------- arg parsing -----------------


def parse_args():
    p = argparse.ArgumentParser("Faster R-CNN folder inference with GT overlay")

    p.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to checkpoint file (state dict or dict with 'model').",
    )
    p.add_argument(
        "--images-dir",
        type=str,
        required=True,
        help="Directory with images to run inference on.",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Where to save visualizations.",
    )
    p.add_argument(
        "--num-classes",
        type=int,
        default=2,
        help="Number of classes (including background).",
    )

    # optional COCO GT
    p.add_argument(
        "--coco-json",
        type=str,
        default=None,
        help="Optional COCO-style JSON to draw ground truth boxes.",
    )

    # thresholds
    p.add_argument(
        "--score-thr",
        type=float,
        default=0.5,
        help="Minimum score to keep a detection.",
    )
    p.add_argument(
        "--nms-iou",
        type=float,
        default=0.4,
        help="IoU threshold for NMS.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Max images to process (0 = all).",
    )

    p.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu'.",
    )
    p.add_argument(
        "--debug-gt",
        action="store_true",
        help="Print how many GT boxes were found per image.",
    )

    return p.parse_args()


# ----------------- main -----------------


def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[info] using device: {device}")

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- build & load model --------
    print("[info] building model...")
    model = build_fasterrcnn(num_classes=args.num_classes, pretrained=False)
    ckpt = torch.load(args.weights, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    print("[info] weights loaded.")

    # -------- optional COCO GT --------
    full_name_to_id = base_name_to_id = annos_by_imgid = catid_to_name = None
    if args.coco_json:
        print(f"[info] loading COCO GT from {args.coco_json}")
        (
            full_name_to_id,
            base_name_to_id,
            annos_by_imgid,
            catid_to_name,
        ) = load_coco_annotations(Path(args.coco_json))

    # -------- gather images --------
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    img_paths = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in exts)
    if args.max_images > 0:
        img_paths = img_paths[: args.max_images]

    print(f"[info] found {len(img_paths)} images.")
    if not img_paths:
        print("[warn] no images found, exiting.")
        return

    # -------- loop over images --------
    for idx, img_path in enumerate(img_paths, 1):
        print(f"[{idx}/{len(img_paths)}] {img_path.name}")

        # load image
        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        # dynamic font size based on image dimensions
        font_size = max(18, min(w, h) // 25)  # tweak 25 for bigger/smaller text
        try:
            font = ImageFont.truetype("arial.ttf", font_size)
        except Exception:
            font = ImageFont.load_default()

        # right panel: original with GT + predictions
        img_viz = img.copy()
        draw = ImageDraw.Draw(img_viz)

        # box thickness scales with image width
        box_width = max(3, w // 400)

        # ----- draw ground truth (green) -----
        if full_name_to_id is not None:
            gt_annos = get_gt_for_image(
                img_path, full_name_to_id, base_name_to_id, annos_by_imgid
            )
            if args.debug_gt:
                print(f"    GT annos: {len(gt_annos)}")
            for ann in gt_annos:
                # COCO bbox: [x, y, w, h]
                x, y, bw, bh = ann["bbox"]
                x2 = x + bw
                y2 = y + bh
                cat_name = catid_to_name.get(ann["category_id"], "gt")

                # draw GT box (green)
                draw.rectangle([x, y, x2, y2], outline="lime", width=box_width)

                # label with black background for readability
                text = f"GT: {cat_name}"
                try:
                    bbox = draw.textbbox((0, 0), text, font=font)
                    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
                except Exception:
                    tw, th = draw.textsize(text, font=font)
                tx, ty = x, max(0, y - th - 4)
                draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill="black")
                draw.text((tx + 2, ty + 2), text, fill="lime", font=font)

        # ----- model prediction (red) -----
        with torch.no_grad():
            im_t = TF.to_tensor(img).unsqueeze(0).to(device)
            out = model(im_t)[0]

        boxes = out["boxes"]
        scores = out["scores"]
        labels = out["labels"]

        # filter by score
        keep = scores >= args.score_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        # NMS
        if boxes.numel() > 0:
            keep_idx = nms(boxes, scores, iou_threshold=args.nms_iou)
            boxes = boxes[keep_idx].cpu()
            scores = scores[keep_idx].cpu()
            labels = labels[keep_idx].cpu()
        else:
            boxes = scores = labels = torch.empty((0,))

        # draw predictions (red)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.tolist()
            draw.rectangle([x1, y1, x2, y2], outline="red", width=box_width)

            txt = f"{score:.2f}"  # or f"pred {int(label.item())}: {score:.2f}"
            try:
                bbox = draw.textbbox((0, 0), text, font=font)
                tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            except Exception:
                tw, th = draw.textsize(text, font=font)
            tx, ty = x1, max(0, y1 - th - 4)
            draw.rectangle([tx, ty, tx + tw + 4, ty + th + 4], fill="black")
            draw.text((tx + 2, ty + 2), txt, fill="red", font=font)

        # side-by-side: left = original, right = GT + preds
        combined = make_side_by_side(img, img_viz)
        out_path = out_dir / f"{img_path.stem}_viz.jpg"
        combined.save(out_path, quality=95)

    print(f"[done] saved visualizations to: {out_dir}")


if __name__ == "__main__":
    main()
