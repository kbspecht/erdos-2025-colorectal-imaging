#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as TF
from tqdm import tqdm

try:
    from IPython.display import display
except Exception:
    display = None


def draw_gt_pred(img, anns, boxes, scores, labels, thr=0.25, font_size=16):
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    try:
        font = ImageFont.truetype("Arial.ttf", font_size)
    except:
        font = ImageFont.load_default()
    for ann in anns:  # GT → green
        x, y, w, h = ann["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="green", width=3)
    for b, s, l in zip(boxes, scores, labels):  # Pred → red
        if float(s) < thr:
            continue
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1 + 4, y1 + 4), f"{float(s):.2f}", fill="red", font=font)
    return vis


def build_model(ckpt, device, num_classes=2):
    m = fasterrcnn_resnet50_fpn_v2(weights=None)
    in_feat = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(
        state if isinstance(state, dict) and "model" not in state else state["model"]
    )
    m.to(device)
    m.eval()
    return m


@torch.no_grad()
def run(model, coco, device, thr, out_dir, show=False, save=True, max_images=None):
    out_dir = Path(out_dir)
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
    img_ids = list(sorted(coco.imgs.keys()))
    if max_images:
        img_ids = img_ids[:max_images]
    for img_id in tqdm(img_ids, desc="viz", ncols=100):
        info = coco.loadImgs([img_id])[0]
        img_path = info["file_name"]
        img = Image.open(img_path).convert("RGB")
        img_t = TF.to_tensor(img).to(device)
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id]))
        pred = model([img_t])[0]
        vis = draw_gt_pred(
            img,
            anns,
            pred["boxes"].cpu(),
            pred["scores"].cpu(),
            pred["labels"].cpu(),
            thr,
        )
        if save:
            vis.save(out_dir / Path(img_path).name)
        if show and display:
            display(vis)
    print(f"[done] {'saved to ' + str(out_dir) if save else 'displayed inline'}")


def parse_args():
    p = argparse.ArgumentParser("Visualize GT vs predictions on COCO val")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--val-json", required=True)
    p.add_argument("--score-thr", type=float, default=0.25)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--out-dir", default="./vis_coco")
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(a.device)
    model = build_model(a.ckpt, device)
    coco = COCO(a.val_json)
    run(
        model,
        coco,
        device,
        a.score_thr,
        a.out_dir,
        show=a.show,
        save=not a.no_save,
        max_images=a.max_images,
    )


if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
