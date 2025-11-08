# This is for the scheduler
import torch.optim as optim

"""
Learning rate scheduler builder for Faster R-CNN training.

- Supports multiple scheduler types:
    • "none"     → no scheduler (returns None)
    • "step"     → StepLR with configurable step_size and gamma
    • "cosine"   → CosineAnnealingLR with configurable T_max
    • "plateau"  → ReduceLROnPlateau (for validation loss monitoring)
- Returns the corresponding PyTorch scheduler object given an optimizer.
- Raises ValueError if an unknown scheduler name is provided.
"""


def build_scheduler(
    name: str,
    optimizer,
    *,
    step_size: int = 5,
    gamma: float = 0.5,
    cosine_tmax: int = 10,
    plateau_factor: float = 0.1,
    plateau_patience: int = 3,
    plateau_threshold: float = 1e-3,
    plateau_min_lr: float = 1e-7,
):

    name = (name or "none").lower()

    if name == "none":
        return None

    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_tmax)

    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            min_lr=plateau_min_lr,  # prints when LR is reduced
        )

    raise ValueError(f"Unknown scheduler: {name}")
