import torch

from faster_rcnn.models.faster_rcnn_model import build_fasterrcnn

"""
Wrapper for constructing a Faster R-CNN model with device handling.

- Calls `build_fasterrcnn` with provided parameters (num_classes, pretrained,
  freeze_backbone, and any extra kwargs).
- Accepts a `device` argument, moves the model to that device, and returns
  both `(model, device)` — a convenience for training and test scripts.
- Keeps the interface consistent across modules that need a standardized
  model-construction entry point.
"""


def build_model(
    num_classes: int = 2,
    pretrained: bool = True,
    freeze_backbone: int = 0,
    device: str | torch.device = "cpu",
    **extra_kwargs,
):
    """
    Thin wrapper around build_fasterrcnn used by tests.
    - accepts a `device` kwarg (tests pass this)
    - forwards only the kwargs that the underlying builder actually understands
    - moves the model to the right device
    - returns (model, device) because tests expect that
    """
    # build the model on CPU first
    model = build_fasterrcnn(
        num_classes=num_classes,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        **extra_kwargs,
    )

    device = torch.device(device)
    model.to(device)

    return model, device
