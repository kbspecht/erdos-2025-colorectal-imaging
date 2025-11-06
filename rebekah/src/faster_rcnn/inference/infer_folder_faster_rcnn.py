# # #!/usr/bin/env python3
# # import argparse
# # from pathlib import Path

# # import torch
# # from PIL import Image, ImageDraw
# # from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# # from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# # from torchvision.transforms import functional as TF
# # from tqdm import tqdm


# # def build_model(num_classes=2, ckpt_path=None, device="cpu"):
# #     m = fasterrcnn_resnet50_fpn_v2(weights=None)
# #     in_feat = m.roi_heads.box_predictor.cls_score.in_features
# #     m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
# #     if ckpt_path:
# #         state = torch.load(str(ckpt_path), map_location=device)
# #         m.load_state_dict(
# #             state
# #             if isinstance(state, dict) and "model" not in state
# #             else state["model"]
# #         )
# #     m.to(device)
# #     m.eval()
# #     return m


# # def load_images(images_dir):
# #     exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# #     return [p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts]


# # @torch.no_grad()
# # def infer_one(model, img_path, device, thr=0.3):
# #     img = Image.open(img_path).convert("RGB")
# #     out = model([TF.to_tensor(img).to(device)])[0]
# #     keep = out["scores"].cpu() >= thr
# #     return (
# #         img,
# #         out["boxes"].cpu()[keep],
# #         out["scores"].cpu()[keep],
# #         out["labels"].cpu()[keep],
# #     )


# # def draw(img, boxes, scores, labels, names=None):
# #     d = ImageDraw.Draw(img)
# #     for b, s, l in zip(boxes, scores, labels):
# #         x1, y1, x2, y2 = b.tolist()
# #         d.rectangle([x1, y1, x2, y2], outline="red", width=3)
# #         cls = str(int(l.item()))
# #         if names and int(l.item()) < len(names):
# #             cls = names[int(l.item())]
# #         d.text((x1 + 2, y1 + 2), f"{cls} {float(s):.2f}", fill="red")
# #     return img


# # def parse_args():
# #     p = argparse.ArgumentParser("Folder inference for Faster R-CNN")
# #     p.add_argument("--ckpt", required=True)
# #     p.add_argument("--images-dir", required=True)
# #     p.add_argument("--out-dir", default="./vis_out")
# #     p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
# #     p.add_argument("--score-thr", type=float, default=0.3)
# #     p.add_argument("--vis", action="store_true")
# #     p.add_argument("--side-by-side", action="store_true")
# #     return p.parse_args()


# # def side_by_side(a, b):
# #     w1, h1 = a.size
# #     w2, h2 = b.size
# #     H = max(h1, h2)
# #     out = Image.new("RGB", (w1 + w2, H), (0, 0, 0))
# #     out.paste(a, (0, 0))
# #     out.paste(b, (w1, 0))
# #     return out


# # def main():
# #     a = parse_args()
# #     device = torch.device(a.device)
# #     model = build_model(num_classes=2, ckpt_path=a.ckpt, device=device))
# #!/usr/bin/env python3
# import argparse
# from pathlib import Path

# import torch
# from PIL import Image, ImageDraw
# from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.transforms import functional as TF
# from tqdm import tqdm


# def build_model(num_classes=2, ckpt_path=None, device="cpu"):
#     m = fasterrcnn_resnet50_fpn_v2(weights=None)
#     in_feat = m.roi_heads.box_predictor.cls_score.in_features
#     m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)

#     if ckpt_path:
#         state = torch.load(str(ckpt_path), map_location=device)
#         # handle both pure state_dict and {"model": state_dict}
#         if isinstance(state, dict) and "model" in state:
#             state = state["model"]
#         m.load_state_dict(state)

#     m.to(device)
#     m.eval()
#     return m


# def load_images(images_dir):
#     exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
#     images_dir = Path(images_dir)
#     return sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])


# @torch.no_grad()
# def infer_one(model, img_path, device, thr=0.3):
#     img = Image.open(img_path).convert("RGB")
#     out = model([TF.to_tensor(img).to(device)])[0]

#     scores = out["scores"].cpu()
#     keep = scores >= thr

#     boxes = out["boxes"].cpu()[keep]
#     scores = scores[keep]
#     labels = out["labels"].cpu()[keep]

#     return img, boxes, scores, labels


# def draw(img, boxes, scores, labels, names=None):
#     img = img.copy()
#     d = ImageDraw.Draw(img)

#     for b, s, l in zip(boxes, scores, labels):
#         x1, y1, x2, y2 = b.tolist()
#         d.rectangle([x1, y1, x2, y2], outline="red", width=3)

#         cls = str(int(l.item()))
#         if names and int(l.item()) in names:
#             cls = names[int(l.item())]

#         text = f"{cls} {float(s):.2f}"
#         d.text((x1 + 2, y1 + 2), text, fill="red")

#     return img


# def side_by_side(a, b):
#     w1, h1 = a.size
#     w2, h2 = b.size
#     H = max(h1, h2)
#     out = Image.new("RGB", (w1 + w2, H), (0, 0, 0))
#     out.paste(a, (0, 0))
#     out.paste(b, (w1, 0))
#     return out


# def parse_args():
#     p = argparse.ArgumentParser("Folder inference for Faster R-CNN")
#     p.add_argument("--ckpt", required=True, help="Path to checkpoint .pth file")
#     p.add_argument("--images-dir", required=True, help="Dir with input images")
#     p.add_argument("--out-dir", default="./vis_out", help="Where to save outputs")
#     p.add_argument(
#         "--device",
#         default="cuda" if torch.cuda.is_available() else "cpu",
#         help="cpu or cuda",
#     )
#     p.add_argument("--score-thr", type=float, default=0.3)
#     p.add_argument(
#         "--side-by-side",
#         action="store_true",
#         help="Save original + prediction side by side",
#     )
#     p.add_argument(
#         "--vis",
#         action="store_true",
#         help="(Optional) Also show images one by one (blocky in scripts)",
#     )
#     return p.parse_args()


# def main():
#     a = parse_args()
#     device = torch.device(a.device)

#     out_dir = Path(a.out_dir)
#     out_dir.mkdir(parents=True, exist_ok=True)

#     print(f"[info] device: {device}")
#     print(f"[info] loading model from: {a.ckpt}")
#     model = build_model(num_classes=2, ckpt_path=a.ckpt, device=device)

#     imgs = load_images(a.images_dir)
#     if not imgs:
#         print(f"[warn] no images found in: {a.images_dir}")
#         return

#     print(f"[info] found {len(imgs)} images in {a.images_dir}")
#     print(f"[info] saving visualizations to {out_dir}")

#     # optional: label names if you want
#     names = {1: "polyp"}

#     for img_path in tqdm(imgs, desc="inference"):
#         img, boxes, scores, labels = infer_one(model, img_path, device, thr=a.score_thr)

#         if len(boxes) == 0:
#             vis = img.copy()
#         else:
#             vis = draw(img, boxes, scores, labels, names=names)

#         if a.side_by_side:
#             vis = side_by_side(img, vis)

#         out_path = out_dir / f"{img_path.stem}_pred.jpg"
#         vis.save(out_path)

#         if a.vis:
#             vis.show()

#     print(f"[done] saved {len(imgs)} images to {out_dir}")


import os

# if __name__ == "__main__":
#     main()
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

#!/usr/bin/env python3
##!/usr/bin/env python3
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms
from torchvision.transforms import functional as TF

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
