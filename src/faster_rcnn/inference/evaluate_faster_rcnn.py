"""
Evaluate a Faster R-CNN checkpoint on a COCO-style validation set.

- Parses CLI args pointing to val.json, roots_map.json, optional images_root,
  model weights, and evaluation hyperparameters.
- Builds a COCO detection dataset with evaluation-time transforms and wraps
  it in a DataLoader.
- Constructs the Faster R-CNN model, loads the checkpoint weights, and
  computes validation loss.
- Optionally runs COCO-style evaluation (coco_map) and prints mAP metrics.
"""

import argparse

import torch
from torch.utils.data import DataLoader

from faster_rcnn.augmentations import build_eval_augs
from faster_rcnn.data.coco_detection_faster_rcnn import CocoDetDataset, collate_fn
from faster_rcnn.data.coco_eval_faster_rcnn import coco_map
from faster_rcnn.models.faster_rcnn_model import build_fasterrcnn
from faster_rcnn.training.engine_faster_rcnn import validate_loss


def parse_args():
    p = argparse.ArgumentParser("Evaluate Faster R-CNN (artifacts/)")
    p.add_argument("--val-json", default="artifacts/val.json")
    p.add_argument("--roots-map", default="artifacts/roots_map.json")
    p.add_argument("--images-root", default=None)  # optional fallback
    p.add_argument("--weights", required=True)

    p.add_argument("--num-classes", type=int, default=2)
    p.add_argument("--img-size", type=int, default=832)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--coco-map", action="store_true")
    return p.parse_args()


def main():
    a = parse_args()
    device = torch.device(
        a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    )

    eval_tf = build_eval_augs(a.img_size)
    val_ds = CocoDetDataset(
        ann_file=a.val_json,
        roots_map_path=a.roots_map,
        images_root=a.images_root,
        transform=eval_tf,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=a.batch_size,
        shuffle=False,
        num_workers=a.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(a.num_workers > 0),
    )

    model = build_fasterrcnn(a.num_classes, pretrained=False).to(device)
    ckpt = torch.load(a.weights, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)

    val_loss = validate_loss(model, val_loader, device)
    print(f"Validation loss: {val_loss:.4f}")

    if a.coco_map:
        stats = coco_map(model, val_loader, device, a.val_json)
        print(
            f"COCO: mAP@[.5:.95]={stats['mAP_50_95']:.4f} | "
            f"mAP@.50={stats['mAP_50']:.4f} | "
            f"precision={stats['precision']:.4f} | "
            f"recall={stats['recall']:.4f}"
        )


if __name__ == "__main__":
    main()
