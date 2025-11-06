from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def _strip_detection_head(state: dict) -> dict:
    """Remove roi_heads predictor weights from a checkpoint so we can load it
    into a model with a different num_classes."""
    to_drop = [
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
    ]
    for k in to_drop:
        state.pop(k, None)
    return state


def build_fasterrcnn(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: int = 0,
    local_weights: str = "artifacts/fasterrcnn_resnet50_fpn_v2.pth",
):
    lw_path = Path(local_weights)

    # 1) base model
    if pretrained and lw_path.exists():
        model = fasterrcnn_resnet50_fpn_v2(weights=None)
        state = torch.load(lw_path, map_location="cpu")

        state = _strip_detection_head(state)

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[info] loaded local weights with missing={len(missing)} unexpected={len(unexpected)}"
            )
        print(f"[info] loaded local Faster R-CNN weights from {lw_path}")
    else:
        model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT" if pretrained else None)

    # 2) replace head with our num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 3) optionally unfreeze last N stages
    if freeze_backbone > 0:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
        stages = [
            model.backbone.body.layer1,
            model.backbone.body.layer2,
            model.backbone.body.layer3,
            model.backbone.body.layer4,
        ]
        for s in stages[-freeze_backbone:]:
            for p in s.parameters():
                p.requires_grad = True

    return model
