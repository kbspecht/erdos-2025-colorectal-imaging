"""
Augmentation utilities for Faster R-CNN style object detection.

Provides:
  - `_to_tensor`: converts numpy images to float32 torch tensors in [0, 1].
  - `_adapter`: wraps Albumentations pipelines to work with torchvision-style
    dict targets (`boxes`, `labels`, `class_labels`, `area`).
  - `build_train_augs`: builds train-time Albumentations pipelines
    ("light", "medium", "strong") with bbox-safe transforms.
  - `build_val_augs`: validation transform that standardizes targets
    without changing image geometry.
"""

from typing import Any, Callable, Dict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

# Optional Albumentations
try:
    import albumentations as A
    import cv2

    _AOK = True
except Exception:
    _AOK = False


def _to_tensor(img: np.ndarray) -> torch.Tensor:
    """np.uint8(H, W, [3]) -> torch.float32(C,H,W) in [0,1]."""
    if img.ndim == 2:  # gray → RGB
        img = np.stack([img] * 3, axis=-1)
    return TF.to_tensor(Image.fromarray(img.astype(np.uint8)))


def _adapter(aug: "A.BasicTransform") -> Callable:
    """
    Wrap an Albumentations pipeline so it accepts:
      (image=np.uint8, target={'boxes','labels' or 'class_labels'})
    and returns:
      (image_tensor, target{'boxes','labels','class_labels','area'})
    Boxes are PASCAL_VOC (x1,y1,x2,y2) in pixels.
    """

    def fn(image: np.ndarray, target: Dict[str, Any]):
        # --- read incoming keys robustly
        boxes = target["boxes"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        boxes = np.asarray(boxes, dtype=np.float32)

        # >>> ensure 4 columns (strip accidental label as 5th col)
        if boxes.ndim == 2 and boxes.shape[1] >= 5:
            boxes = boxes[:, :4]

        # Prefer 'class_labels' (matches BboxParams), fallback to 'labels'
        in_labels = target.get("class_labels", target.get("labels", []))
        if isinstance(in_labels, torch.Tensor):
            in_labels = in_labels.cpu().numpy()
        in_labels = np.asarray(in_labels, dtype=np.int64).tolist()

        res = aug(
            image=image,
            bboxes=boxes.tolist(),
            class_labels=in_labels,
        )

        img2 = res["image"]
        bb = np.asarray(res.get("bboxes", []), dtype=np.float32)
        ll = np.asarray(res.get("class_labels", []), dtype=np.int64)

        # Drop degenerate boxes defensively
        keep = [i for i, (x1, y1, x2, y2) in enumerate(bb) if x2 > x1 and y2 > y1]
        if keep:
            bb = bb[keep]
            ll = ll[keep]
        else:
            bb = np.zeros((0, 4), np.float32)
            ll = np.zeros((0,), np.int64)

        # Post-clip to image bounds (handles tiny negatives after warps)
        H, W = img2.shape[:2]
        if bb.size:
            x1 = np.clip(bb[:, 0], 0, W - 1)
            y1 = np.clip(bb[:, 1], 0, H - 1)
            x2 = np.clip(bb[:, 2], 0, W - 1)
            y2 = np.clip(bb[:, 3], 0, H - 1)
            bb = np.stack([x1, y1, x2, y2], axis=1)

        # Update target (torchvision expects 'labels')
        target["boxes"] = torch.as_tensor(bb, dtype=torch.float32)
        target["labels"] = torch.as_tensor(ll, dtype=torch.int64)
        target["class_labels"] = target["labels"]

        # Keep 'area' consistent after aug
        if bb.shape[0] > 0:
            areas = (bb[:, 2] - bb[:, 0]) * (bb[:, 3] - bb[:, 1])
            target["area"] = torch.as_tensor(areas, dtype=torch.float32)
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        return _to_tensor(img2), target

    return fn


def build_train_augs(img_size: int = 832, preset: str = "light") -> Callable:
    """
    Returns a callable: (image np.uint8, target dict) -> (tensor, target).
    If Albumentations is unavailable or preset=='none', returns identity-ish.
    """
    if not _AOK or preset == "none":
        return lambda image, target: (_to_tensor(image), target)

    def pad_block():
        return A.PadIfNeeded(
            min_height=img_size,
            min_width=img_size,
            border_mode=cv2.BORDER_CONSTANT,
            fill=0,  # <- 'fill' is the new arg (not 'value')
        )

    if preset == "light":
        aug = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                pad_block(),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.15),
                A.Affine(  # replaces ShiftScaleRotate
                    scale=(0.9, 1.1),
                    translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                    rotate=(-10, 10),
                    cval=0,
                    fit_output=False,
                ),
                A.RandomBrightnessContrast(p=0.25),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.0,
                clip=True,  # <- clip coords into image
            ),
        )

    elif preset == "medium":
        aug = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                pad_block(),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.8, 1.2),
                    translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                    rotate=(-15, 15),
                    cval=0,
                    fit_output=False,
                ),
                A.RandomBrightnessContrast(p=0.30),
                A.GaussNoise(p=0.20),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.0,
                clip=True,
            ),
        )

    else:  # strong
        aug = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                pad_block(),
                A.HorizontalFlip(p=0.5),
                A.Affine(
                    scale=(0.7, 1.3),
                    translate_percent={"x": (-0.08, 0.08), "y": (-0.08, 0.08)},
                    rotate=(-20, 20),
                    cval=0,
                    fit_output=False,
                ),
                A.RandomBrightnessContrast(p=0.40),
                A.MotionBlur(p=0.20),
                A.GaussNoise(p=0.20),
                A.RandomGamma(p=0.20),
            ],
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["class_labels"],
                min_visibility=0.0,
                clip=True,
            ),
        )

    return _adapter(aug)


def build_val_augs(img_size: int = 832) -> Callable:
    """
    Validation transforms: no geometric changes, just:
      - ensure tensor image
      - ensure target has 'boxes', 'labels', 'class_labels', 'area'

    We deliberately avoid Albumentations here so that validation images
    stay in their original resolution (matching COCO GT JSON).
    """

    def fn(image: np.ndarray, target: Dict[str, Any]):
        # boxes may be tensor or np; keep as tensor in xyxy
        boxes = target["boxes"]
        if isinstance(boxes, torch.Tensor):
            bb = boxes.clone().to(dtype=torch.float32)
        else:
            bb = torch.as_tensor(boxes, dtype=torch.float32)

        # get labels from either 'labels' or 'class_labels'
        labels = target.get("labels", target.get("class_labels", None))
        if labels is None:
            raise KeyError("Expected 'labels' or 'class_labels' in target for val.")
        if isinstance(labels, torch.Tensor):
            ll = labels.clone().to(dtype=torch.int64)
        else:
            ll = torch.as_tensor(labels, dtype=torch.int64)

        # update target fields in the format torchvision expects
        target["boxes"] = bb
        target["labels"] = ll
        target["class_labels"] = ll  # keep both for consistency

        # recompute area
        if bb.numel() > 0:
            x1, y1, x2, y2 = bb.unbind(-1)
            areas = (x2 - x1) * (y2 - y1)
            target["area"] = areas
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        # DO NOT change geometry of the image – just to_tensor
        image_t = _to_tensor(image)
        return image_t, target

    return fn
