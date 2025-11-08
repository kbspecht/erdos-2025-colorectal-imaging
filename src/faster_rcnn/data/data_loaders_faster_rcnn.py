# Augmentations
from torch.utils.data import DataLoader

from faster_rcnn.augmentations.presets_faster_rcnn import (
    build_train_augs,
    build_val_augs,
)
from faster_rcnn.data.coco_detection_faster_rcnn import CocoDetDataset, collate_fn


def build_train_val_loaders(
    train_json: str,
    val_json: str,
    roots_map: str,
    images_root: str | None,
    img_size: int,
    train_augs: str,
    batch_size: int,
    num_workers: int,
):

    train_tf = build_train_augs(img_size, train_augs)
    eval_tf = build_val_augs(img_size)

    # Datasets / DataLoaders (consume artifacts and roots_map)
    train_ds = CocoDetDataset(
        ann_file=train_json,
        roots_map_path=roots_map,
        images_root=images_root,
        transform=train_tf,
    )
    val_ds = CocoDetDataset(
        ann_file=val_json,
        roots_map_path=roots_map,
        images_root=images_root,
        transform=eval_tf,
    )

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)

    return train_loader, val_loader
