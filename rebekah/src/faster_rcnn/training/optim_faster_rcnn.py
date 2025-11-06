# optim_faster_rcnn.py
from typing import Dict, List

import torch
import torch.optim as optim


def _split_param_groups(model, lr_backbone, lr_heads, weight_decay, tiered: bool):
    """
    Returns param groups for optimizer.
    - If tiered=False: 2 groups → backbone (ResNet body + FPN) vs heads (RPN + ROI).
    - If tiered=True:  3 groups → body vs FPN vs heads.
    Includes only params with requires_grad=True.
    """
    groups: List[Dict] = []

    if tiered:
        body = [p for _, p in model.backbone.body.named_parameters() if p.requires_grad]
        fpn = [p for _, p in model.backbone.fpn.named_parameters() if p.requires_grad]
        heads = []
        heads += [p for _, p in model.rpn.named_parameters() if p.requires_grad]
        heads += [p for _, p in model.roi_heads.named_parameters() if p.requires_grad]

        if body:
            groups.append(
                {"params": body, "lr": lr_backbone * 0.5, "weight_decay": weight_decay}
            )  # e.g., slower
        if fpn:
            groups.append(
                {"params": fpn, "lr": lr_backbone, "weight_decay": weight_decay}
            )
        if heads:
            groups.append(
                {"params": heads, "lr": lr_heads, "weight_decay": weight_decay}
            )
        return groups

    # 2-tier default: backbone vs heads
    backbone = [p for _, p in model.backbone.named_parameters() if p.requires_grad]
    heads = []
    heads += [p for _, p in model.rpn.named_parameters() if p.requires_grad]
    heads += [p for _, p in model.roi_heads.named_parameters() if p.requires_grad]

    if backbone:
        groups.append(
            {"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay}
        )
    if heads:
        groups.append({"params": heads, "lr": lr_heads, "weight_decay": weight_decay})
    return groups


def build_optimizer(
    model: torch.nn.Module,
    opt: str = "sgd",
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    # new (optional) knobs:
    lr_backbone: float | None = None,
    lr_heads: float | None = None,
    tiered: bool = False,
):
    """
    If lr_backbone/lr_heads are provided → param groups with distinct LRs.
    Otherwise falls back to a single-lr optimizer over all trainable params.
    """
    opt = opt.lower()

    # If user wants differential LRs:
    if (lr_backbone is not None) and (lr_heads is not None):
        param_groups = _split_param_groups(
            model,
            lr_backbone=lr_backbone,
            lr_heads=lr_heads,
            weight_decay=weight_decay,
            tiered=tiered,
        )
        if opt == "sgd":
            return optim.SGD(param_groups, momentum=momentum)
        if opt == "adam":
            return optim.Adam(param_groups)
        if opt == "adamw":
            return optim.AdamW(param_groups)
        raise ValueError(f"Unknown optimizer: {opt}")

    # Fallback: single-lr over all trainable params (your original behavior)
    params = [p for p in model.parameters() if p.requires_grad]
    if opt == "sgd":
        return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    if opt == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if opt == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {opt}")
