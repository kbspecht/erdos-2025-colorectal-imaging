#!/usr/bin/env python3
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


def _anns_to_voc(anns: List[Dict[str, Any]]):
    boxes, labels, areas, iscrowd = [], [], [], []
    for a in anns:
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            continue
        boxes.append([x, y, x + w, y + h])
        labels.append(a.get("category_id", 1))
        areas.append(a.get("area", w * h))
        iscrowd.append(a.get("iscrowd", 0))
    if not boxes:
        return (
            np.zeros((0, 4), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            np.zeros((0,), np.int64),
        )
    return (
        np.asarray(boxes, np.float32),
        np.asarray(labels, np.int64),
        np.asarray(areas, np.float32),
        np.asarray(iscrowd, np.int64),
    )


class CocoDetDataset(Dataset):
    """
    COCO detection dataset with robust path resolution for your artifacts/ layout.

    Resolution order for image path:
      1) If COCO 'file_name' is absolute, use it.
      2) Else, look up basename in roots_map (basename -> abs path).
      3) Else, if images_root was provided, join it with file_name.
    """

    def __init__(
        self,
        ann_file: str,
        roots_map_path: Optional[str] = None,
        images_root: Optional[str] = None,
        transform=None,
    ):
        super().__init__()
        self.coco = COCO(ann_file)
        self.ids = list(self.coco.imgs.keys())
        self.transform = transform
        self.images_root = images_root

        # Load roots_map.json if provided
        self.roots_map = {}
        if roots_map_path and os.path.isfile(roots_map_path):
            with open(roots_map_path, "r") as f:
                self.roots_map = json.load(f)

    def __len__(self) -> int:
        return len(self.ids)

    def _resolve_path(self, file_name: str) -> str:
        # 1) absolute → use as-is
        if os.path.isabs(file_name):
            return file_name
        # 2) basename lookup in roots_map
        base = os.path.basename(file_name)
        if base in self.roots_map:
            return self.roots_map[base]
        # 3) optional images_root fallback
        if self.images_root is not None:
            return os.path.join(self.images_root, file_name)
        # last resort: return as-is (will error if unreadable)
        return file_name

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        info = self.coco.loadImgs([img_id])[0]
        img_path = self._resolve_path(info["file_name"])
        image = np.array(Image.open(img_path).convert("RGB"))

        ann_ids = self.coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        anns = self.coco.loadAnns(ann_ids)
        boxes, labels, areas, iscrowd = _anns_to_voc(anns)

        target = {
            "boxes": torch.from_numpy(boxes),
            "labels": torch.from_numpy(labels),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "area": torch.from_numpy(areas),
            "iscrowd": torch.from_numpy(iscrowd),
        }

        if self.transform is not None:
            # albumentations adapter expects 'class_labels'
            image_t, upd = self.transform(
                image=image,
                target={"boxes": target["boxes"], "class_labels": target["labels"]},
            )
            target["boxes"], target["labels"] = upd["boxes"], upd["labels"]
            image = image_t
        else:
            image = TF.to_tensor(Image.fromarray(image))
        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)
