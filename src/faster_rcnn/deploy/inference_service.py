# inference_service.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import torch
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from faster_rcnn.models.faster_rcnn_model import build_fasterrcnn

# Only needed if you actually use YOLO


# Type for model family
ModelKind = Literal["fasterrcnn", "yolo"]

# Root of your repo inside the container
REPO_ROOT = Path(__file__).resolve().parents[2]  # e.g. /app/src/... -> /app


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where we expect checkpoints to live
FASTER_DIR = REPO_ROOT / "runs" / "frcnn_polyp"
YOLO_DIR = REPO_ROOT / "runs" / "yolo"


# MODEL LOADING


def _load_fasterrcnn(ckpt_path: str | Path | None):
    """
    Load your Faster R-CNN model from a .pth checkpoint.
    """
    if ckpt_path is None:
        ckpt_path = FASTER_DIR / "best.pth"

    ckpt_path = Path(ckpt_path)

    model = build_fasterrcnn(num_classes=2)  # adjust if your builder differs
    state = torch.load(ckpt_path, map_location=DEVICE)

    # adjust to how you saved
    state_dict = (
        state["model"] if isinstance(state, dict) and "model" in state else state
    )
    model.load_state_dict(state_dict)

    model.to(DEVICE)
    model.eval()
    return model


def _load_yolo(ckpt_path: str | Path | None):
    """
    Load a YOLO model via ultralytics.
    """
    if YOLO is None:
        raise RuntimeError(
            "YOLO requested but the 'ultralytics' package is not installed."
        )

    if ckpt_path is None:
        ckpt_path = YOLO_DIR / "best.pt"

    ckpt_path = Path(ckpt_path)
    # YOLO handles device internally; you can still force .to(DEVICE) if needed.
    model = YOLO(str(ckpt_path))
    return model


def load_model(kind: ModelKind, ckpt_path: str | Path | None = None):
    """
    Unified loader for different model families.

    Parameters
    ----------
    kind : {"fasterrcnn", "yolo"}
        Which type of model to load.
    ckpt_path : str | Path | None
        Path to the checkpoint to use. If None, a default is used per family.
    """
    if kind == "fasterrcnn":
        return _load_fasterrcnn(ckpt_path)
    elif kind == "yolo":
        return _load_yolo(ckpt_path)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


# -------------------------------------------------------------------
# INFERENCE ADAPTERS
# -------------------------------------------------------------------


def _run_inference_fasterrcnn(model, image_pil: Image.Image, score_thr: float):
    """
    Run inference for Faster R-CNN and return a normalized list of predictions.
    """
    import torchvision.transforms.functional as TF

    img_t = TF.to_tensor(image_pil).to(DEVICE).unsqueeze(0)  # [1, C, H, W]

    with torch.no_grad():
        outputs = model(img_t)[0]

    boxes = outputs["boxes"].cpu().numpy()  # (N, 4) xyxy
    scores = outputs["scores"].cpu().numpy()  # (N,)

    preds: list[dict] = []
    for i, (box, score) in enumerate(zip(boxes, scores)):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box
        preds.append(
            {
                "id": i,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
            }
        )
    return preds


def _run_inference_yolo(model, image_pil: Image.Image, score_thr: float):
    """
    Run inference for YOLO (Ultralytics) and return normalized predictions.

    Assumes:
      results = model(image_pil, conf=score_thr)[0]
      results.boxes.xyxy → (N, 4), results.boxes.conf → (N,)
    """
    # conf=score_thr already filters low scores, but we still check
    results = model(image_pil, conf=score_thr, verbose=False)[0]

    boxes_xyxy = results.boxes.xyxy.cpu().numpy()  # (N, 4)
    scores = results.boxes.conf.cpu().numpy()  # (N,)

    preds: list[dict] = []
    for i, (box, score) in enumerate(zip(boxes_xyxy, scores)):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box
        preds.append(
            {
                "id": i,
                "box": [float(x1), float(y1), float(x2), float(y2)],
                "score": float(score),
            }
        )
    return preds


def run_inference(
    kind: ModelKind,
    model,
    image_pil: Image.Image,
    score_thr: float = 0.3,
):
    """
    Unified inference API for any supported model family.

    Returns
    -------
    preds : list of dict
        Each dict has keys: "id", "box" (xyxy), "score".
    """
    if kind == "fasterrcnn":
        return _run_inference_fasterrcnn(model, image_pil, score_thr)
    elif kind == "yolo":
        return _run_inference_yolo(model, image_pil, score_thr)
    else:
        raise ValueError(f"Unknown model kind: {kind}")


# -------------------------------------------------------------------
# VISUALIZATION + GROUND TRUTH (unchanged from your version)
# -------------------------------------------------------------------


def draw_boxes(image_pil: Image.Image, preds: list[dict]):
    """
    Draw boxes with ID and score, using larger text size.
    Works on Pillow >=10 (no .textsize()) and earlier versions.
    """
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    # pick a larger font (auto-scale to image width)
    font_size = max(16, int(image_pil.width / 50))
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for p in preds:
        x1, y1, x2, y2 = p["box"]
        score = p["score"]
        det_id = p.get("id", None)

        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        label = f"#{det_id} ({score:.2f})" if det_id is not None else f"{score:.2f}"

        # determine text width/height safely
        if hasattr(font, "getbbox"):  # Pillow ≥10
            left, top, right, bottom = font.getbbox(label)
            tw, th = right - left, bottom - top
        else:  # fallback for older Pillow
            tw, th = font.getsize(label)

        # background rectangle for visibility
        text_x, text_y = x1 + 3, max(y1 - th - 4, 0)
        draw.rectangle(
            [text_x, text_y, text_x + tw + 4, text_y + th + 2],
            fill="white",
        )

        # draw the text itself
        draw.text((text_x + 2, text_y), label, fill="red", font=font)

    return img


def load_ground_truth(gt_json_path: str | Path, image_name: str):
    """
    Return list of GT boxes in xyxy format for a given image filename
    from COCO-style JSON.
    """
    gt_json_path = Path(gt_json_path)
    with gt_json_path.open("r") as f:
        data = json.load(f)

    id_by_name = {img["file_name"]: img["id"] for img in data["images"]}

    gt_boxes: list[list[float]] = []
    img_id = id_by_name.get(image_name)
    if img_id is None:
        return gt_boxes

    for ann in data["annotations"]:
        if ann["image_id"] == img_id:
            x, y, w, h = ann["bbox"]
            gt_boxes.append([x, y, x + w, y + h])  # xywh -> xyxy
    return gt_boxes


def draw_boxes_with_gt(
    image_pil: Image.Image,
    preds: list[dict],
    gt_boxes: list[list[float]],
):
    """
    Overlay model predictions (red) and GT boxes (green).
    """
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)

    # predicted boxes in red
    for p in preds:
        x1, y1, x2, y2 = p["box"]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        draw.text((x1, y1), f"{p['score']:.2f}", fill="red")

    # ground truth boxes in green
    for x1, y1, x2, y2 in gt_boxes:
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=3)

    return img


# cd /Users/rebekaheichberg/Desktop/git-re/colorectal_imaging/Architectures
# PYTHONPATH=src:$PYTHONPATH streamlit run src/faster_rcnn/deploy/streamlit_app.py
