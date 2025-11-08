import os
import random
from dataclasses import dataclass

import numpy as np
import torch

"""
Utility helpers for Faster R-CNN training and experiments.

- `set_seed`: seeds Python, NumPy, and PyTorch (CPU + CUDA), and optionally
  enforces deterministic cuDNN behavior.
- `setup_perf`: enables TF32 (when available) for faster matmuls and tweaks
  an env var for smoother host→device behavior.
- `ensure_dir`: creates a directory path if it doesn’t already exist.
- `AverageMeter`: tracks running average of a metric (e.g., loss) over steps.
- `EarlyStopping`: monitors a metric to stop training when it stops improving
  for a given patience (minimization mode, e.g., val_loss).
"""


def set_seed(seed: int = 1337, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


def setup_perf(tf32: bool = True):

    if tf32 and hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Faster host->device transfers
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.n = 0

    def update(self, v, k=1):
        self.sum += float(v) * k
        self.n += k

    @property
    def avg(self):
        return self.sum / max(1, self.n)


@dataclass
class EarlyStoppingState:
    best: float = float("inf")
    epochs_no_improve: int = 0


class EarlyStopping:
    """Minimize monitored metric (e.g., val_loss)."""

    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.state = EarlyStoppingState()

    def step(self, value: float) -> bool:
        if value < (self.state.best - self.min_delta):
            self.state.best = value
            self.state.epochs_no_improve = 0
            return False
        self.state.epochs_no_improve += 1
        return self.state.epochs_no_improve > self.patience
