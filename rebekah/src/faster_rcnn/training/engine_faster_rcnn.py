import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from faster_rcnn.utils.utils_faster_rcnn import AverageMeter, ensure_dir


def train_one_epoch(
    model, loader: DataLoader, optimizer, scaler, device, epoch, log_every=50
):
    model.train()
    loss_meter = AverageMeter()
    for it, (images, targets) in enumerate(
        tqdm(loader, desc=f"Train E{epoch}", ncols=100)
    ):
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss_dict = model(images, targets)  # dict of losses
            loss = sum(loss_dict.values())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), k=len(images))
        if (it + 1) % log_every == 0:
            print(f"[E{epoch} I{it+1}] train_loss={loss_meter.avg:.4f}")
    return loss_meter.avg


@torch.no_grad()
def validate_loss(model, loader: DataLoader, device) -> float:
    """
    To get a loss dict, torchvision detection models must be in train() mode.
    Use no_grad() so weights don't update.
    """
    model.train()
    meter = AverageMeter()
    for images, targets in loader:
        images = [im.to(device) for im in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)  # returns dict in train()
        loss = sum(loss_dict.values())
        meter.update(loss.item(), k=len(images))
    model.eval()
    return meter.avg


def save_checkpoint(
    state: dict, out_dir: str, is_best: bool = False, filename="last.pth"
):
    ensure_dir(out_dir)
    torch.save(state, f"{out_dir}/{filename}")
    if is_best:
        torch.save(state, f"{out_dir}/best.pth")
    if is_best:
        torch.save(state, f"{out_dir}/best.pth")
