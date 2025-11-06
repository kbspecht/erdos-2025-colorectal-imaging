#!/usr/bin/env python3
"""
Download Faster R-CNN ResNet50 FPN v2 weights once and save locally.

Run:
    python download_frcnn_weights.py --out artifacts/fasterrcnn_resnet50_fpn_v2.pth
"""

import argparse
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2


def parse_args():
    p = argparse.ArgumentParser("Download Faster R-CNN weights")
    p.add_argument(
        "--out",
        type=str,
        default="artifacts/fasterrcnn_resnet50_fpn_v2.pth",
        help="Where to save the downloaded state_dict",
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        help="Load model on CPU (safer on machines without CUDA)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = (
        torch.device("cpu")
        if args.cpu or not torch.cuda.is_available()
        else torch.device("cuda")
    )
    print(f"[info] loading model on {device} (this may download the weights)...")

    # this line is what actually triggers the download
    model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT").to(device)
    sd = model.state_dict()

    torch.save(sd, out_path)
    print(f"[done] saved model state_dict → {out_path.resolve()}")

    # optional: tell user about the torch cache
    torch_cache = Path(torch.hub.get_dir())
    print(f"[info] torch hub cache is at: {torch_cache.resolve()}")


if __name__ == "__main__":
    main()
