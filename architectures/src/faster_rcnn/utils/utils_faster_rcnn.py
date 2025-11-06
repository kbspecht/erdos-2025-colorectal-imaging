import os, random
import numpy as np
import torch
from dataclasses import dataclass

def set_seed(seed: int = 1337, deterministic: bool = False):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic

def setup_perf(tf32: bool = True):
    # 4070 supports TF32; this speeds up matmuls while preserving numerics for detection
    if tf32 and hasattr(torch.backends, "cuda") and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Faster host->device transfers
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum = 0.0; self.n = 0
    def update(self, v, k=1): self.sum += float(v) * k; self.n += k
    @property
    def avg(self): return self.sum / max(1, self.n)

@dataclass
class EarlyStoppingState:
    best: float = float("inf")
    epochs_no_improve: int = 0

class EarlyStopping:
    """Minimize monitored metric (e.g., val_loss)."""
    def __init__(self, patience=8, min_delta=0.0):
        self.patience = patience; self.min_delta = min_delta
        self.state = EarlyStoppingState()
    def step(self, value: float) -> bool:
        if value < (self.state.best - self.min_delta):
            self.state.best = value; self.state.epochs_no_improve = 0
            return False
        self.state.epochs_no_improve += 1
        return self.state.epochs_no_improve > self.patience
