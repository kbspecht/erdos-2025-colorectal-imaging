from ultralytics import YOLO
from ultralytics.utils import LOGGER
import logging
import numpy as np
from pathlib import Path
import base64
import csv
from IPython.display import HTML, display, Image

# -------------------------
# Logging (quiet Ultralytics)
# -------------------------
logging.basicConfig(level=logging.INFO)
# Set Ultralytics to WARNING level to suppress most output
logging.getLogger("ultralytics").setLevel(logging.WARNING)

# -------------------------
# Helpers
# -------------------------
def _scalar(x):
    """Return a single float from scalar/array metrics (macro average for arrays)."""
    try:
        return float(x)
    except Exception:
        arr = np.asarray(x)
        if arr.size == 1:
            return float(arr.reshape(()))
        return float(np.nanmean(arr))


def _b64_img(path: Path) -> str | None:
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def _read_agg_from_results_csv(save_dir: Path) -> tuple[int | None, int | None]:
    """Try to read Images/Instances from results.csv 'all' row."""
    csv_path = Path(save_dir) / "results.csv"
    if not csv_path.exists():
        return None, None
    try:
        with open(csv_path, newline="") as f:
            lines = f.readlines()
            if not lines:
                return None, None
            
            header = [h.strip() for h in lines[0].split(",")]
            reader = csv.DictReader(lines[1:], fieldnames=header)
            
            for row in reader:
                if row.get("class", "").strip().lower() == "all":
                    images = row.get("images")
                    instances = row.get("instances")
                    
                    img_n = int(float(images)) if images not in (None, "") else None
                    ins_n = int(float(instances)) if instances not in (None, "") else None
                    return img_n, ins_n
    except Exception:
        pass
    return None, None


def _get_images_instances(metrics) -> tuple[int | None, int | None]:
    """Return (images, instances) by accessing the Validator object, with CSV fallback."""
    
    # --- 1. Primary Method: Accessing the Validator object ---
    validator = getattr(metrics.model, 'validator', None)
    
    n_images = None
    n_instances = None
    
    if validator is not None:
        n_images = getattr(validator, 'n_images', None)
        if n_images is None and hasattr(validator, 'dataloader') and validator.dataloader is not None:
             n_images = len(validator.dataloader.dataset)
        
        if getattr(validator, 'stats', None) is not None and len(validator.stats) > 3:
            n_instances = int(validator.stats[3])
            
        if n_instances is None and hasattr(validator, 'confusion_matrix'):
            try:
                n_instances = int(validator.confusion_matrix.matrix.sum(axis=1).sum())
            except Exception:
                pass
                
    # --- 2. Fallback to direct attributes (less reliable) ---
    if n_images is None:
        n_images = getattr(metrics, "images", None)
    
    if n_instances is None:
        n_instances = getattr(metrics, "nboxes", None) or getattr(metrics, "seen", None)

    # --- 3. Fallback to results.csv (least reliable) ---
    if n_images is None or n_instances is None:
        csv_imgs, csv_insts = _read_agg_from_results_csv(Path(metrics.save_dir))
        n_images = n_images if n_images is not None else csv_imgs
        n_instances = n_instances if n_instances is not None else csv_insts
        
    return n_images, n_instances


# -------------------------
# Main evaluation function
# -------------------------
def evaluate_yolo_model(
    model_path: str | Path,
    data_yaml: str | Path,
    split: str = "test",
    save_dir: str | Path = "runs/test",
):
    """
    Evaluate a YOLO model (detection or segmentation) on a given split and
    write a concise Markdown report with dataset stats, metrics, and plots.

    Args:
        model_path: Path to YOLO .pt file.
        data_yaml:  Path to data YAML file.
        split:      Dataset split ('train'|'val'|'test'). Default: 'test'.
        save_dir:   Directory where Ultralytics outputs and Markdown are saved.
                    Final files under: <parent(save_dir)>/<name(save_dir)>/
    """
    model_path = Path(model_path)
    data_yaml = Path(data_yaml)
    model_name = model_path.stem
    save_dir = Path(save_dir).resolve()

    # Run evaluation; place outputs to save_dir's parent/name
    model = YOLO(model_path)
    
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        plots=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        verbose=False,
    )
    
    # Attach the model (and thus the validator) to the metrics object for helper access
    metrics.model = model 

    # Markdown summary
    report_root = Path(metrics.save_dir) / "markdown"
    report_root.mkdir(parents=True, exist_ok=True)

    md_lines = [
        f"## Evaluation Results — {model_name}",
        "",
        f"Split: `{split}`  ",
        f"Results folder: `{metrics.save_dir}`  ",
        "",
    ]

    # Dataset stats: Number of images / Number of polyp instances
    img_n, ins_n = _get_images_instances(metrics)
    if img_n is not None or ins_n is not None:
        md_lines += [
            "### Dataset Stats",
            "| Stat | Value |",
            "|---|---:|",
        ]
        if img_n is not None:
            md_lines.append(f"| Number of images | {img_n} |")
        if ins_n is not None:
            md_lines.append(f"| Number of polyp instances | {ins_n} |")
        md_lines.append("")

    # Detection metrics (if available)
    if hasattr(metrics, "box"):
        box = metrics.box
        md_lines += [
            "### Detection Metrics",
            "| Metric | Value |",
            "|---|---:|",
            f"| Precision | {_scalar(box.p):.3f} |",
            f"| Recall | {_scalar(box.r):.3f} |",
            f"| mAP@50 | {_scalar(box.map50):.3f} |",
            f"| mAP@50-95 | {_scalar(box.map):.3f} |",
            "",
        ]

    # Segmentation metrics (if available)
    if hasattr(metrics, "seg"):
        seg = metrics.seg
        md_lines += [
            "### Segmentation Metrics",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mask Precision | {_scalar(seg.p):.3f} |",
            f"| Mask Recall | {_scalar(seg.r):.3f} |",
            f"| mAP mask@50 | {_scalar(seg.map50):.3f} |",
            f"| mAP mask@50-95 | {_scalar(seg.map):.3f} |",
            "",
        ]

    # Add plots (exact names if present)
    plot_dir = Path(metrics.save_dir)
    plot_names = ["results.png", "BoxPR_curve.png", "MaskPR_curve.png", "PR_curve.png", "confusion_matrix.png"]
    for p in plot_names:
        if (plot_dir / p).exists():
            md_lines.append(f"![{p}]({plot_dir}/{p})")

    report_file = report_root / f"{model_name}_REPORT.md"
    report_file.write_text("\n".join(md_lines), encoding="utf-8")
    print(f"Report saved to: {report_file}")

    return metrics


# -------------------------
# Notebook display helper 
# -------------------------
def show_detection_summary(
    metrics,
    scale: float = 0.33,
    plot_names: tuple[str, str, str] = None,
    debug: bool = False,
):
    """
    Print dataset stats and detection/segmentation metrics; show relevant plots.

    Args:
        metrics: result of YOLO(...).val(...)
        scale:   fractional width per image (0.5 for 2 plots, 0.33 for 3 plots)
        plot_names: preferred plot filenames (up to 3); will fall back to common names
        debug:   if True, list all PNGs in the results folder
    """
    plot_dir = Path(metrics.save_dir)
    is_segmentation = hasattr(metrics, "seg")
    is_detection = hasattr(metrics, "box")
    
    # --- Print Dataset Stats ---
    img_n, ins_n = _get_images_instances(metrics)
    if img_n is not None:
        print(f"Number of images: {img_n}")
    if ins_n is not None:
        print(f"Number of polyp instances: {ins_n}")

    # --- Print Metrics ---
    if is_detection:
        box = metrics.box
        print("\n**Detection Metrics (Box):**")
        print(f" Precision:  {_scalar(box.p):.3f}")
        print(f" Recall:     {_scalar(box.r):.3f}")
        print(f" mAP@50:     {_scalar(box.map50):.3f}")
        print(f" mAP@50-95:  {_scalar(box.map):.3f}")
    
    if is_segmentation:
        seg = metrics.seg
        print("\n**Segmentation Metrics (Mask):**")
        print(f" Precision:  {_scalar(seg.p):.3f}")
        print(f" Recall:     {_scalar(seg.r):.3f}")
        print(f" mAP mask@50:  {_scalar(seg.map50):.3f}")
        print(f" mAP mask@50-95:  {_scalar(seg.map):.3f}")

    if not is_detection and not is_segmentation:
        print("No detection or segmentation metrics found on this metrics object.")

    # --- Resolve Plots (Collect plots based on task) ---
    pr_plots = []
    cm_plot = None
    
    # 1. Prioritize based on user input (if provided)
    if plot_names and len(plot_names) > 0:
        plot_paths = [plot_dir / name for name in plot_names]
        imgs = [p for p in plot_paths if p.exists()]
    else:
        # 2. Dynamic collection for Segmentation (Box PR, Mask PR, Confusion Matrix)
        if is_segmentation:
            # Collect Box PR Curve
            box_pr_candidates = ["BoxPR_curve.png", "PR_curve.png"]
            box_pr_path = next((plot_dir / c for c in box_pr_candidates if (plot_dir / c).exists()), None)
            if box_pr_path:
                pr_plots.append(box_pr_path)
                
            # Collect Mask PR Curve
            mask_pr_candidates = ["MaskPR_curve.png"]
            mask_pr_path = next((plot_dir / c for c in mask_pr_candidates if (plot_dir / c).exists()), None)
            if mask_pr_path:
                pr_plots.append(mask_pr_path)
                
            # Collect Confusion Matrix
            cm_candidates = ["confusion_matrix.png", "confusion_matrix_normalized.png"]
            cm_plot = next((plot_dir / c for c in cm_candidates if (plot_dir / c).exists()), None)
        
        # 3. Dynamic collection for Detection Only (Box PR, Confusion Matrix)
        elif is_detection:
            # Collect single PR Curve (Box)
            pr_candidates = ["BoxPR_curve.png", "PR_curve.png"]
            pr_path = next((plot_dir / c for c in pr_candidates if (plot_dir / c).exists()), None)
            if pr_path:
                pr_plots.append(pr_path)
                
            # Collect Confusion Matrix
            cm_candidates = ["confusion_matrix.png", "confusion_matrix_normalized.png"]
            cm_plot = next((plot_dir / c for c in cm_candidates if (plot_dir / c).exists()), None)
            
        # Combine all found plots
        imgs = pr_plots
        if cm_plot:
            imgs.append(cm_plot)

    if not imgs:
        print("No plots found to display. Set debug=True to list available images.")
        return

    # --- Side-by-side display via base64 (Robust in notebooks) ---
    try:
        # Determine the percentage width for each plot
        num_plots = len(imgs)
        # Calculate the base percentage width, and subtract a small margin (e.g., 2%) for spacing
        base_width = 100 / num_plots 
        margin_adjust = 2 * (num_plots - 1) / num_plots # Distribute 2% of total width as margin
        default_pct_width = int(base_width - margin_adjust)
        
        # Use user-provided scale if it's smaller, otherwise use the calculated default
        width_pct = int(scale * 100) if scale and scale * 100 < default_pct_width else default_pct_width

        html = f"<div style='display:flex; gap:8px; align-items:flex-start; max-width:100%;'>"
        for p in imgs:
            b64 = _b64_img(p)
            if not b64:
                continue
            # The key fix: set flex-basis to force the column width to the calculated percentage
            html += (
                f"<div style='flex:0 0 {width_pct}%;'>"
                f"<img src='data:image/png;base64,{b64}' style='width:100%; height:auto;'/>"
                f"<div style='font-size:12px; color:#555; text-align:center;'>{p.name}</div>"
                f"</div>"
            )
        html += "</div>"
        display(HTML(html))
    except Exception as e:
        print("Display fallback:", e)
        # Fallback to standard Image display
        try:
            for p in imgs:
                display(Image(filename=str(p)))
        except Exception as e2:
            print("Could not display images:", e2)
            for p in imgs:
                print(p)