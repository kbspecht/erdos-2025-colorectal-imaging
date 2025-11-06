# This is for the scheduler
import torch.optim as optim


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

    if name == "pleateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=plateau_factor,
            patience=plateau_patience,
            threshold=plateau_threshold,
            min_lr=plateau_min_lr,  # prints when LR is reduced
        )

    raise ValueError(f"Unknown scheduler: {name}")
