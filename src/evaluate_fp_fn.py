"""
evaluate_fp_fn.py
Evaluate a YOLO (seg or det) model on a folder of images + YOLO labels, and
return False Positives, True Positives, and False Negatives with useful metadata
(box area, ratio to image, image size), ready for downstream analysis.

Usage (from notebooks):
-----------------------
from src.evaluate_fp_fn import evaluate_model_predictions

res = evaluate_model_predictions(
    model_path=r"C:\...\best.pt",
    labels_dir=r"C:\...\val\labels",
    images_dir=r"C:\...\val\images",
    conf_thresh=0.001,   # optional
    iou_thresh=0.5       # optional
)

fp, tp, fn = res["false_positives"], res["true_positives"], res["false_negatives"]
"""

from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from math import ceil


# ------------------ small utilities ------------------

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for [x1,y1,x2,y2] boxes in pixels."""
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    areaA = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    areaB = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = areaA + areaB - inter
    return float(inter / union) if union > 0 else 0.0

def box_area_xyxy(box: np.ndarray) -> float:
    """Area of [x1,y1,x2,y2] in pixels."""
    return float(max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1]))

def area_ratio(area_px: float, img_w: int, img_h: int) -> float:
    """Box area / image area."""
    denom = float(max(1, img_w * img_h))
    return float(area_px / denom)

def _load_image_size(img_path: Path) -> Tuple[int, int]:
    im = cv2.imread(str(img_path))
    if im is None:
        raise FileNotFoundError(f"Failed to read image: {img_path}")
    h, w = im.shape[:2]
    return w, h


# ------------------ GT loader (det or seg) ------------------

def load_gt_boxes(img_path: Path, labels_dir: Path) -> List[Dict]:
    """
    Load GT boxes from YOLO detection or segmentation labels for one image.
    Supports:
      - det: <cls> cx cy w h
      - seg: <cls> x1 y1 x2 y2 ... (polygon), normalized in [0,1]
    Returns list of dicts with:
      { "cls": int, "box": [x1,y1,x2,y2], "area": float, "ratio": float, "img_w": int, "img_h": int }
    """
    lab_path = labels_dir / (img_path.stem + ".txt")
    if not lab_path.exists():
        return []

    img_w, img_h = _load_image_size(img_path)
    gts: List[Dict] = []

    with open(lab_path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            p = ln.strip().split()
            if len(p) < 5:
                continue

            try:
                cls_id = int(float(p[0]))
            except:
                # tolerate malformed class field
                continue

            vals = list(map(float, p[1:]))

            # pure detection line (cx,cy,w,h normalized)
            if len(vals) == 4:
                cx, cy, w, h = vals
                x1 = (cx - w / 2.0) * img_w
                y1 = (cy - h / 2.0) * img_h
                x2 = (cx + w / 2.0) * img_w
                y2 = (cy + h / 2.0) * img_h

            # segmentation polygon line (x1,y1,x2,y2,... normalized)
            else:
                if len(vals) % 2 == 1:
                    # drop last odd value defensively
                    vals = vals[:-1]
                xs = np.array(vals[0::2], dtype=float)
                ys = np.array(vals[1::2], dtype=float)
                if xs.size == 0 or ys.size == 0:
                    continue
                x1, y1 = xs.min() * img_w, ys.min() * img_h
                x2, y2 = xs.max() * img_w, ys.max() * img_h

            box = np.array([x1, y1, x2, y2], dtype=float)
            area = box_area_xyxy(box)
            ratio = area_ratio(area, img_w, img_h)

            gts.append({
                "cls": cls_id,
                "box": box.tolist(),
                "area": area,
                "ratio": ratio,
                "img_w": img_w,
                "img_h": img_h
            })

    return gts


# ------------------ main evaluation ------------------

def evaluate_model_predictions(
    model_path: str,
    labels_dir: str,
    images_dir: str,
    conf_thresh: float = None,
    iou_thresh: float = 0.5
) -> Dict[str, List[Dict]]:
    """
    Run model inference on images_dir and compare to YOLO labels in labels_dir.
    Returns:
      {
        "false_positives": [ {image, img_w, img_h, pred_box, pred_area, pred_ratio, conf, cls} ... ],
        "true_positives":  [ {image, img_w, img_h, pred_box, pred_area, pred_ratio, conf, cls,
                               gt_box, gt_area, gt_ratio, gt_cls, iou} ... ],
        "false_negatives": [ {image, img_w, img_h, gt_box, gt_area, gt_ratio, gt_cls,
                               predictions_on_image=[{pred_box, conf, cls}, ...]} ... ]
      }
    Notes:
      - Works with seg or det GT labels (see load_gt_boxes).
      - Predictions are taken from model.predict (NMS already applied by Ultralytics).
    """
    model = YOLO(model_path)
    labels_dir = Path(labels_dir)
    images_dir = Path(images_dir)

    fps: List[Dict] = []
    tps: List[Dict] = []
    fns: List[Dict] = []

    image_paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    for img_path in image_paths:
        # ground truth
        gt_list = load_gt_boxes(img_path, labels_dir)

        # predictions (boxes in xyxy pixel space)
        if conf_thresh is not None:
            r = model.predict(str(img_path), verbose=False, conf=conf_thresh)[0]
        else:
            r = model.predict(str(img_path), verbose=False)[0]

        if r.boxes is None or len(r.boxes) == 0:
            pred_boxes  = np.empty((0, 4), dtype=float)
            pred_confs  = np.empty((0,), dtype=float)
            pred_classes= np.empty((0,), dtype=float)
        else:
            pred_boxes   = r.boxes.xyxy.cpu().numpy().astype(float)
            pred_confs   = r.boxes.conf.cpu().numpy().astype(float)
            pred_classes = r.boxes.cls.cpu().numpy().astype(float)

        # image size
        img_w, img_h = _load_image_size(img_path)

        # match predictions -> GT (greedy by best IoU)
        matched_gt = set()
        for j, pb in enumerate(pred_boxes):
            best_i, best_k = 0.0, -1
            for k, gt in enumerate(gt_list):
                if k in matched_gt:
                    continue
                i = iou_xyxy(pb, np.asarray(gt["box"], dtype=float))
                if i > best_i:
                    best_i, best_k = i, k

            pred_area = box_area_xyxy(pb)
            pred_ratio = area_ratio(pred_area, img_w, img_h)

            if best_i >= iou_thresh and best_k >= 0:
                gt = gt_list[best_k]
                tps.append({
                    "image": img_path.name,
                    "img_w": img_w, "img_h": img_h,
                    "pred_box": pb.tolist(),
                    "pred_area": float(pred_area),
                    "pred_ratio": float(pred_ratio),
                    "conf": float(pred_confs[j]),
                    "cls": int(pred_classes[j]) if pred_classes.size else 0,
                    "gt_box": gt["box"],
                    "gt_area": float(gt["area"]),
                    "gt_ratio": float(gt["ratio"]),
                    "gt_cls": int(gt["cls"]),
                    "iou": float(best_i)
                })
                matched_gt.add(best_k)
            else:
                fps.append({
                    "image": img_path.name,
                    "img_w": img_w, "img_h": img_h,
                    "pred_box": pb.tolist(),
                    "pred_area": float(pred_area),
                    "pred_ratio": float(pred_ratio),
                    "conf": float(pred_confs[j]),
                    "cls": int(pred_classes[j]) if pred_classes.size else 0,
                })

        # any GT not matched -> FN
        for k, gt in enumerate(gt_list):
            if k not in matched_gt:
                fn_item = {
                    "image": img_path.name,
                    "img_w": img_w, "img_h": img_h,
                    "gt_box": gt["box"],
                    "gt_area": float(gt["area"]),
                    "gt_ratio": float(gt["ratio"]),
                    "gt_cls": int(gt["cls"]),
                }

                # NEW: include any predictions that were made on this image
                if len(pred_boxes) > 0:
                    preds_for_image = []
                    for j in range(len(pred_boxes)):
                        preds_for_image.append({
                            "pred_box": pred_boxes[j].tolist(),
                            "conf": float(pred_confs[j]),
                            "cls": int(pred_classes[j]) if pred_classes.size else 0,
                        })
                    fn_item["predictions_on_image"] = preds_for_image
                else:
                    fn_item["predictions_on_image"] = []

                fns.append(fn_item)

    # simple summary
    print(f"\n✅ Done on {len(image_paths)} images.")
    print(f"TP={len(tps)}  FP={len(fps)}  FN={len(fns)}")

    return {
        "false_positives": fps,
        "true_positives": tps,
        "false_negatives": fns,
    }

# ------------------ comparison utilities ------------------

def compare_detection_results(res1: Dict[str, List[Dict]], 
                              res2: Dict[str, List[Dict]], 
                              iou_thresh: float = 0.5) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Compare FP/TP/FN results from two models.

    Args:
        res1, res2: dicts returned by evaluate_model_predictions()
        iou_thresh: IoU threshold to treat detections as 'same object'

    Returns:
        {
          "true_positives": {
              "intersection": [...],   # boxes detected by both
              "unique_1": [...],       # unique to model1
              "unique_2": [...],       # unique to model2
          },
          "false_positives": {... same ...},
          "false_negatives": {... same ...},
        }
    """
    def _match_sets(list1, list2):
        inter, only1, only2 = [], [], []
        matched2 = set()

        for i, a in enumerate(list1):
            box_a = np.array(a.get("pred_box") or a.get("gt_box"), dtype=float)
            img_a = a["image"]
            found = False
            for j, b in enumerate(list2):
                if j in matched2 or img_a != b["image"]:
                    continue
                box_b = np.array(b.get("pred_box") or b.get("gt_box"), dtype=float)
                if iou_xyxy(box_a, box_b) >= iou_thresh:
                    inter.append(a)
                    matched2.add(j)
                    found = True
                    break
            if not found:
                only1.append(a)

        # collect items from list2 that were unmatched
        for j, b in enumerate(list2):
            if j not in matched2:
                only2.append(b)

        return {"intersection": inter, "unique_1": only1, "unique_2": only2}

    return {
        "true_positives":  _match_sets(res1["true_positives"],  res2["true_positives"]),
        "false_positives": _match_sets(res1["false_positives"], res2["false_positives"]),
        "false_negatives": _match_sets(res1["false_negatives"], res2["false_negatives"]),
    }


# =======================
# SINGLE-MODEL DASHBOARD
# =======================

def plot_counts_bar(res: dict, label: str = "Model"):
    tp, fp, fn = len(res["true_positives"]), len(res["false_positives"]), len(res["false_negatives"])
    cats = ["TP", "FP", "FN"]
    vals = [tp, fp, fn]
    plt.figure(figsize=(4, 3), dpi=120)
    bars = plt.bar(cats, vals)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width()/2, b.get_height(), str(v),
                 ha="center", va="bottom", fontsize=9)
    plt.title(f"{label}: TP / FP / FN counts", fontsize=10)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()

def _image_area_array(items):
    # items each have img_w, img_h
    if not items: return np.array([])
    return np.array([float(d["img_w"]) * float(d["img_h"]) for d in items], dtype=float)

def plot_kind_distribution(res: dict, kind: str, label: str = "Model"):
    """
    Plot side-by-side (smaller) histograms for a single kind:
      kind in {'tp','fp','fn'}
      Left  = polyp relative area
      Right = image area (log-x if all > 0)
    """
    import numpy as np
    import matplotlib.pyplot as plt

    assert kind in {"tp","fp","fn"}, "kind must be one of {'tp','fp','fn'}"

    if kind == "tp":
        items = res["true_positives"]
        rel = np.array([d.get("gt_ratio", 0.0) for d in items], dtype=float)
        title_kind = "True Positives"
        color = "#4daf4a"
    elif kind == "fp":
        items = res["false_positives"]
        rel = np.array([d.get("pred_ratio", 0.0) for d in items], dtype=float)
        title_kind = "False Positives"
        color = "#e41a1c"
    else:
        items = res["false_negatives"]
        rel = np.array([d.get("gt_ratio", 0.0) for d in items], dtype=float)
        title_kind = "False Negatives"
        color = "#377eb8"

    imgA = _image_area_array(items)
    n = len(items)

    # Smaller figure, two plots on one row
    fig, axes = plt.subplots(1, 2, figsize=(10, 3), dpi=120, constrained_layout=True)
    ax1, ax2 = axes

    # Relative area
    ax1.hist(rel, color=color, alpha=0.85, rwidth=0.9)
    ax1.set_xlabel("Relative area (box_area / image_area)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"{label} — {title_kind}: polyp size (n={n})", fontsize=10)
    ax1.grid(alpha=0.3)

    # Image area
    ax2.hist(imgA, color=color, alpha=0.85, rwidth=0.9)
    if np.all(imgA > 0):
        ax2.set_xscale("log")
    ax2.set_xlabel("Image area (pixels²)")
    ax2.set_ylabel("Count")
    ax2.set_title(f"{label} — {title_kind}: image size (n={n})", fontsize=10)
    ax2.grid(alpha=0.3)

    plt.show()

def plot_size_and_image_distributions(res: dict, label: str = "Model",
                                      ratio_bins=None, img_area_bins=None):
    """
    Backward-compatible wrapper that now shows compact side-by-side histograms for TP, FP, FN.
    """
    plot_kind_distribution(res, "tp", label==label)
    plot_kind_distribution(res, "fp", label=label)
    plot_kind_distribution(res, "fn", label=label)

def single_model_dashboard(res: dict, label: str = "Model",
                           ratio_bins=None, img_area_bins=None):
    """Counts bar + compact polyp-size & image-size distributions for TP, FP, FN."""
    plot_counts_bar(res, label=label)
    plot_size_and_image_distributions(res, label=label)


# =======================
# COMPARISON (DIFFERENCES)
# =======================

def _items_to_arrays(items, *, kind: str):
    """
    Convert a list of detection dicts to arrays:
      - polyp relative area (FP->pred_ratio, FN->gt_ratio)
      - image area
    kind: 'fp' or 'fn'
    """
    rel = []
    imgA = []
    key = "pred_ratio" if kind == "fp" else "gt_ratio"
    for d in items:
        rel.append(d.get(key, 0.0))
        imgA.append(float(d["img_w"]) * float(d["img_h"]))
    return np.array(rel, float), np.array(imgA, float)

def _bar_with_counts(ax, cats, vals, title=None, ylabel=None):
    bars = ax.bar(cats, vals)
    for b, v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height(), str(int(v)),
                ha="center", va="bottom", fontsize=9)
    if title: ax.set_title(title, fontsize=10)
    if ylabel: ax.set_ylabel(ylabel)

def compare_differences_dashboard(res1: dict, res2: dict,
                                  label1: str = "Model 1", label2: str = "Model 2",
                                  iou_thresh: float = 0.5, ratio_bins=None, img_area_bins=None):
    """
    Show difference analysis (unique FP/FN per model) with compact, side-by-side histograms.
    """
    import numpy as np
    # compute uniques via your existing set matcher
    cmp = compare_detection_results(res1, res2, iou_thresh=iou_thresh)

    # ---- FPs: uniques
    fp1_only = cmp["false_positives"]["unique_1"]
    fp2_only = cmp["false_positives"]["unique_2"]
    fp1_rel, fp1_imgA = _items_to_arrays(fp1_only, kind="fp")
    fp2_rel, fp2_imgA = _items_to_arrays(fp2_only, kind="fp")

    # ---- FNs: uniques
    fn1_only = cmp["false_negatives"]["unique_1"]
    fn2_only = cmp["false_negatives"]["unique_2"]
    fn1_rel, fn1_imgA = _items_to_arrays(fn1_only, kind="fn")
    fn2_rel, fn2_imgA = _items_to_arrays(fn2_only, kind="fn")

    # --- FP block: sizes (compact)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=120, constrained_layout=True)
    _bar_with_counts(axes[0], [f"{label1} only", f"{label2} only"],
                     [len(fp1_only), len(fp2_only)],
                     title="FP differences (unique)", ylabel="Count")

    axes[1].hist(fp1_rel, color="#e41a1c", alpha=0.85, rwidth=0.9)
    axes[1].set_xlabel("Rel. area")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label1} FP-only size (n={len(fp1_rel)})", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].hist(fp2_rel, color="#377eb8", alpha=0.85, rwidth=0.9)
    axes[2].set_xlabel("Rel. area")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"{label2} FP-only size (n={len(fp2_rel)})", fontsize=10)
    axes[2].grid(alpha=0.3)
    plt.show()

    # --- FP block: image sizes (compact)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=120, constrained_layout=True)
    _bar_with_counts(axes[0], [f"{label1} only", f"{label2} only"],
                     [len(fp1_only), len(fp2_only)],
                     title="FP differences (unique)", ylabel="Count")

    axes[1].hist(fp1_imgA, color="#e41a1c", alpha=0.85, rwidth=0.9)
    if np.all(fp1_imgA > 0): axes[1].set_xscale("log")
    axes[1].set_xlabel("Image area (px²)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label1} FP-only image size (n={len(fp1_imgA)})", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].hist(fp2_imgA, color="#377eb8", alpha=0.85, rwidth=0.9)
    if np.all(fp2_imgA > 0): axes[2].set_xscale("log")
    axes[2].set_xlabel("Image area (px²)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"{label2} FP-only image size (n={len(fp2_imgA)})", fontsize=10)
    axes[2].grid(alpha=0.3)
    plt.show()

    # --- FN block: sizes (compact)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=120, constrained_layout=True)
    _bar_with_counts(axes[0], [f"{label1} only", f"{label2} only"],
                     [len(fn1_only), len(fn2_only)],
                     title="FN differences (unique)", ylabel="Count")

    axes[1].hist(fn1_rel, color="#e41a1c", alpha=0.85, rwidth=0.9)
    axes[1].set_xlabel("Rel. area")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label1} FN-only size (n={len(fn1_rel)})", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].hist(fn2_rel, color="#377eb8", alpha=0.85, rwidth=0.9)
    axes[2].set_xlabel("Rel. area")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"{label2} FN-only size (n={len(fn2_rel)})", fontsize=10)
    axes[2].grid(alpha=0.3)
    plt.show()

    # --- FN block: image sizes (compact)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3), dpi=120, constrained_layout=True)
    _bar_with_counts(axes[0], [f"{label1} only", f"{label2} only"],
                     [len(fn1_only), len(fn2_only)],
                     title="FN differences (unique)", ylabel="Count")

    axes[1].hist(fn1_imgA, color="#e41a1c", alpha=0.85, rwidth=0.9)
    if np.all(fn1_imgA > 0): axes[1].set_xscale("log")
    axes[1].set_xlabel("Image area (px²)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"{label1} FN-only image size (n={len(fn1_imgA)})", fontsize=10)
    axes[1].grid(alpha=0.3)

    axes[2].hist(fn2_imgA, color="#377eb8", alpha=0.85, rwidth=0.9)
    if np.all(fn2_imgA > 0): axes[2].set_xscale("log")
    axes[2].set_xlabel("Image area (px²)")
    axes[2].set_ylabel("Count")
    axes[2].set_title(f"{label2} FN-only image size (n={len(fn2_imgA)})", fontsize=10)
    axes[2].grid(alpha=0.3)
    plt.show()

    # Return lists so you can also feed them to example viewers
    return {
        "fp": {"unique_1": fp1_only, "unique_2": fp2_only},
        "fn": {"unique_1": fn1_only, "unique_2": fn2_only},
        "cmp": cmp
    }


# ==========================
# EXAMPLES (WITH DIFFERENCES)
# ==========================

def _draw_box(ax, box, color="r", lw=2, label=None):
    x1, y1, x2, y2 = [float(v) for v in box]
    ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                               fill=False, linewidth=lw, edgecolor=color))
    if label:
        ax.text(x1, max(0, y1 - 3), label, fontsize=8, color=color,
                bbox=dict(facecolor="black", alpha=0.3, pad=1))

def show_examples(res: dict, images_dir: str, labels_dir: str,
                  kind: str = "fp", n: int = 9, sort_by: str = None):
    """
    Show examples directly from a results dict returned by evaluate_model_predictions().
    kind ∈ {'fp','fn','tp'}.
    """
    images_dir = Path(images_dir); labels_dir = Path(labels_dir)
    assert kind in {"fp","fn","tp"}

    items = (res["false_positives"] if kind == "fp"
             else res["false_negatives"] if kind == "fn"
             else res["true_positives"])

    if not items:
        print(f"No items for kind={kind}."); return

    # default sort
    if sort_by is None:
        sort_by = "conf" if kind == "fp" else ("gt_area" if kind == "fn" else "iou")
    items = sorted(items, key=lambda d: d.get(sort_by, 0.0), reverse=True)[:n]

    # More columns + smaller figure
    cols = 4 if len(items) >= 8 else 3
    rows = ceil(len(items) / cols)
    plt.figure(figsize=(cols * 3.0, rows * 3.0), dpi=120)

    for i, d in enumerate(items):
        img_path = images_dir / d["image"]
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(im); ax.axis("off")

        title = f"{kind.upper()} • {d['image']}"
        if kind == "fp":
            _draw_box(ax, d["pred_box"], color="r", lw=2,
                      label=f"pred conf={d.get('conf', 0):.2f}")
            # faint GTs for context
            for gt in load_gt_boxes(img_path, labels_dir):
                _draw_box(ax, gt["box"], color="cyan", lw=1)
        elif kind == "fn":
            # Missed GT (cyan)
            _draw_box(ax, d["gt_box"], color="cyan", lw=2, label="missed GT")
            # NEW: draw any predictions that exist on this image (red, with conf)
            for p in d.get("predictions_on_image", []):
                _draw_box(ax, p["pred_box"], color="r", lw=2,
                          label=f"pred conf={p.get('conf', 0):.2f}")
        else:  # tp
            _draw_box(ax, d["gt_box"],  color="cyan", lw=2, label="GT")
            _draw_box(ax, d["pred_box"], color="r",   lw=2,
                      label=f"pred conf={d.get('conf', 0):.2f}")
            title += f" • IoU={d.get('iou', 0):.2f}"

        ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.show()

def show_examples_from_items(items: list, images_dir: str, labels_dir: str,
                             kind: str = "fp", n: int = 9, sort_by: str = None):
    """
    Same viewer but takes a LIST of items (e.g., uniques from compare_detection_results()).
    kind ∈ {'fp','fn','tp'} to decide what to draw.
    """
    images_dir = Path(images_dir); labels_dir = Path(labels_dir)
    assert kind in {"fp","fn","tp"}

    if not items:
        print(f"No items to display for kind={kind}."); return

    if sort_by is None:
        sort_by = "conf" if kind == "fp" else ("gt_area" if kind == "fn" else "iou")
    items = sorted(items, key=lambda d: d.get(sort_by, 0.0), reverse=True)[:n]

    cols = 4 if len(items) >= 8 else 3
    rows = ceil(len(items) / cols)
    plt.figure(figsize=(cols * 3.0, rows * 3.0), dpi=120)

    for i, d in enumerate(items):
        img_path = images_dir / d["image"]
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(im); ax.axis("off")

        title = f"{kind.upper()} • {d['image']}"
        if kind == "fp":
            _draw_box(ax, d["pred_box"], color="r", lw=2,
                      label=f"pred conf={d.get('conf', 0):.2f}")
            for gt in load_gt_boxes(img_path, labels_dir):
                _draw_box(ax, gt["box"], color="cyan", lw=1)
        elif kind == "fn":
            _draw_box(ax, d["gt_box"], color="cyan", lw=2, label="missed GT")
            # NEW: draw any predictions stored on this FN item
            for p in d.get("predictions_on_image", []):
                _draw_box(ax, p["pred_box"], color="r", lw=2,
                          label=f"pred conf={p.get('conf', 0):.2f}")
        else:  # tp
            _draw_box(ax, d["gt_box"],  color="cyan", lw=2, label="GT")
            _draw_box(ax, d["pred_box"], color="r",   lw=2,
                      label=f"pred conf={d.get('conf', 0):.2f}")
            title += f" • IoU={d.get('iou', 0):.2f}"

        ax.set_title(title, fontsize=8)
    plt.tight_layout()
    plt.show()
