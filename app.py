# import json
# from datetime import datetime
# from pathlib import Path

# import pandas as pd
# import streamlit as st
# from PIL import Image

# from src.faster_rcnn.deploy.inference_service import (
#     ModelKind,
#     draw_boxes,
#     load_model,
#     run_inference,
# )

# # PATHS
# REPO_ROOT = Path(__file__).resolve().parent

# FEEDBACK_DIR = REPO_ROOT / "feedback"
# FEEDBACK_DIR.mkdir(exist_ok=True)
# FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

# MODEL_DIRS: dict[str, Path] = {
#     "fasterrcnn": REPO_ROOT / "models" / "faster_rcnn",
#     "yolo": REPO_ROOT / "models" / "yolo",
# }
# DEMO_IMAGES_DIR = REPO_ROOT / "demo_images"


# def save_feedback(record: dict) -> None:
#     """Append one feedback record as a JSON line."""
#     with FEEDBACK_FILE.open("a") as f:
#         f.write(json.dumps(record) + "\n")


# def list_checkpoints(kind: ModelKind):
#     """
#     Return a sorted list of checkpoints for a given model family.
#     Faster R-CNN → *.pth, YOLO → *.pt
#     """
#     model_dir = MODEL_DIRS[kind]
#     if not model_dir.exists():
#         return []

#     pattern = "*.pth" if kind == "fasterrcnn" else "*.pt"
#     return sorted(model_dir.glob(pattern))


# @st.cache_resource
# def get_model(kind: str, ckpt_path: str):
#     """
#     Cached model loader. Cache key is (kind, ckpt_path),
#     so switching either reloads once and then reuses.
#     """
#     return load_model(kind, ckpt_path)


# # STREAMLIT APP


# def main():
#     st.title("Polyp Detection – Patient Review Demo")

#     # ----------------- MODEL SELECTION (SIDEBAR) -----------------

#     # 1) Choose model family
#     family_label_to_kind: dict[str, ModelKind] = {
#         "Faster R-CNN": "fasterrcnn",
#         "YOLO": "yolo",
#     }

#     family_label = st.sidebar.selectbox(
#         "Model Architecture",
#         options=list(family_label_to_kind.keys()),
#     )
#     model_kind: ModelKind = family_label_to_kind[family_label]

#     # 2) Choose checkpoint for that family
#     ckpt_files = list_checkpoints(model_kind)
#     if not ckpt_files:
#         st.error(
#             f"No checkpoints found for `{model_kind}` in `{MODEL_DIRS[model_kind]}`.\n\n"
#             "Make sure your trained models (e.g. best.pth / best.pt, epoch_XXX.pth / .pt) "
#             "are saved there."
#         )
#         return

#     # Default selection: use best.* if present, else last in sorted list
#     default_idx = 0
#     for i, p in enumerate(ckpt_files):
#         if p.name.lower().startswith("best"):
#             default_idx = i
#             break
#     else:
#         default_idx = len(ckpt_files) - 1

#     selected_ckpt = st.sidebar.selectbox(
#         "Model checkpoint",
#         options=ckpt_files,
#         index=default_idx,
#         format_func=lambda p: p.name,
#     )

#     st.sidebar.caption(f"Using {family_label} checkpoint: `{selected_ckpt.name}`")

#     # 3) Confidence threshold
#     score_thr = st.sidebar.slider(
#         "Confidence threshold",
#         min_value=0.1,
#         max_value=0.9,
#         value=0.3,
#         step=0.05,
#     )
#     st.sidebar.write("This threshold adjusts the confidence of each prediction.")
#     st.sidebar.markdown(
#         """
# **Threshold note**

# Only detections with a model score **≥ this value** are shown.

# - Lower it to see **more** boxes (higher sensitivity, more false positives)
# - Raise it to see **fewer** boxes (higher precision, more missed polyps)
# """
#     )

#     # ----------------- PATIENT + IMAGE INPUT -----------------

#     patient_id = st.text_input("Patient ID (optional)", value="")

#     uploaded_files = st.file_uploader(
#         "Upload all colonoscopy frames for this patient.",
#         type=["jpg", "jpeg", "png"],
#         accept_multiple_files=True,
#     )

#     if not uploaded_files:
#         st.info("Upload one or more images to get started.")
#         return

#     num_images = len(uploaded_files)
#     # ---- Determine series composition ----
#     seq_files = [f for f in uploaded_files if "seq" in f.name.lower()]
#     single_files = [f for f in uploaded_files if "seq" not in f.name.lower()]

#     if seq_files and single_files:
#         series_type = (
#             f"Mixed upload: {len(seq_files)} sequence frame(s) "
#             f"and {len(single_files)} single frame(s)"
#         )
#     elif seq_files:
#         series_type = f"There are {len(seq_files)} sequence frames."
#     else:
#         series_type = f"There are {len(single_files)} single frame images."

#     st.markdown(
#         f"**Patient summary:** {num_images} image(s) uploaded. "
#         f"Series type: `{series_type}`."
#     )

#     # Frame selection
#     idx = st.selectbox(
#         "Select frame to review",
#         options=list(range(num_images)),
#         format_func=lambda i: f"{i + 1}/{num_images} – {uploaded_files[i].name}",
#     )

#     selected_file = uploaded_files[idx]
#     image_pil = Image.open(selected_file).convert("RGB")
#     img_name = selected_file.name

#     # ----------------- RUN MODEL -----------------

#     model = get_model(model_kind, str(selected_ckpt))
#     preds = run_inference(model_kind, model, image_pil, score_thr=score_thr)

#     # ----------------- DISPLAY IMAGES -----------------

#     st.subheader("Original Image and Model Predictions")
#     col1, col2 = st.columns(2)

#     with col1:
#         st.markdown("**Original Image**")
#         st.image(
#             image_pil,
#             caption=f"{img_name} (frame {idx + 1}/{num_images})",
#             use_container_width=True,
#         )

#     with col2:
#         st.markdown("**Predictions (red boxes)**")
#         img_with_boxes = draw_boxes(image_pil, preds)
#         st.image(
#             img_with_boxes,
#             caption=f"Model predictions – {family_label}",
#             use_container_width=True,
#         )

#     st.write(f"Found {len(preds)} detections with score ≥ {score_thr:.2f}.")

#     # ----------------- DETECTIONS TABLE -----------------

#     if preds:
#         st.subheader("Detections summary")
#         df = pd.DataFrame(
#             [
#                 {
#                     "Polyp ID": p["id"],
#                     "Probability of true polyp": round(p["score"], 3),
#                 }
#                 for p in preds
#             ]
#         )
#         st.dataframe(df, use_container_width=True, hide_index=True)
#     else:
#         st.info("No detections above the threshold.")

#     # ----------------- DOCTOR REVIEW / FEEDBACK -----------------

#     st.subheader("Doctor review")
#     feedback_entries: list[dict] = []

#     if preds:
#         for p in preds:
#             det_id = p["id"]
#             score = p["score"]
#             box = p["box"]

#             with st.expander(f"Detection #{det_id} (score={score:.2f})", expanded=True):
#                 st.write(
#                     f"Box: x1={int(box[0])}, y1={int(box[1])}, "
#                     f"x2={int(box[2])}, y2={int(box[3])}"
#                 )

#                 marked_incorrect = st.checkbox(
#                     "Mark this detection as incorrect (false positive)",
#                     key=f"incorrect_{idx}_{det_id}",  # include frame index in key
#                     value=False,
#                 )

#                 note = st.text_area(
#                     "Optional note (e.g. 'fold', 'bubble', 'normal mucosa')",
#                     key=f"note_{idx}_{det_id}",
#                     height=60,
#                 )

#                 feedback_entries.append(
#                     {
#                         "det_id": det_id,
#                         "box": box,
#                         "score": score,
#                         "marked_incorrect": marked_incorrect,
#                         "note": note.strip(),
#                     }
#                 )
#     else:
#         st.info("No detections above the threshold to review.")

#     # Missed polyp feedback
#     st.subheader("Missed polyps")
#     missed_note = st.text_area(
#         "If the model missed a polyp in this frame, describe where it is relative to the others.",
#         key=f"missed_note_{idx}",
#         height=80,
#     )

#     # ----------------- SAVE FEEDBACK -----------------

#     if st.button("Save feedback for this frame"):
#         record = {
#             "timestamp": datetime.utcnow().isoformat(),
#             "patient_id": patient_id,
#             "num_images_for_patient": num_images,
#             "frame_index": idx,
#             "image_name": img_name,
#             "score_threshold": score_thr,
#             "model_family": model_kind,
#             "model_checkpoint": selected_ckpt.name,
#             "feedback_per_detection": feedback_entries,
#             "missed_polyps_note": missed_note.strip(),
#         }
#         save_feedback(record)
#         st.success("Feedback saved for this frame.")
#         st.write(f"Appended to {FEEDBACK_FILE}")


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
"""
Streamlit app for polyp detection review.

- Lets the user choose a model family (Faster R-CNN or YOLO) and a checkpoint.
- Supports two image sources:
    * Built-in demo images from `demo_images/`
    * User-uploaded colonoscopy frames
- Runs inference on a selected frame and displays:
    * Original image
    * Predictions overlaid as red boxes
    * A table of detections with probabilities
- Collects doctor feedback (false positives, notes, and missed polyps)
  and appends it as JSONL records under `feedback/feedback.jsonl`.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st
from PIL import Image

from src.faster_rcnn.deploy.inference_service import (
    ModelKind,
    draw_boxes,
    load_model,
    run_inference,
)

# -------------------------------------------------------------------
# PATHS / CONSTANTS
# -------------------------------------------------------------------

# This file lives at the repo root: ERDOS-2025-COLORECTAL-IMAGING/app.py
REPO_ROOT = Path(__file__).resolve().parent

FEEDBACK_DIR = REPO_ROOT / "feedback"
FEEDBACK_DIR.mkdir(parents=True, exist_ok=True)
FEEDBACK_FILE = FEEDBACK_DIR / "feedback.jsonl"

# Where model checkpoints live
MODEL_DIRS: Dict[ModelKind, Path] = {
    "fasterrcnn": REPO_ROOT / "models" / "faster_rcnn",
    "yolo": REPO_ROOT / "models" / "yolo",
}

# Built-in demo images to make the app usable out-of-the-box
DEMO_IMAGES_DIR = REPO_ROOT / "demo_images"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# -------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------


def save_feedback(record: dict) -> None:
    """Append one feedback record as a JSON line."""
    with FEEDBACK_FILE.open("a") as f:
        f.write(json.dumps(record) + "\n")


def _glob_images(folder: Path) -> List[Path]:
    """Return sorted list of image paths under a folder."""
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def list_checkpoints(kind: ModelKind):
    """
    Return a sorted list of checkpoints for a given model family.

    Faster R-CNN → *.pth (under models/faster_rcnn)
    YOLO        → *.pt  (under models/)
    """
    model_dir = MODEL_DIRS[kind]
    if not model_dir.exists():
        return []

    pattern = "*.pth" if kind == "fasterrcnn" else "*.pt"
    return sorted(model_dir.glob(pattern))


# @st.cache_resource
def get_model(kind: ModelKind, ckpt_path: str):
    """
    Cached model loader. Cache key is (kind, ckpt_path),
    so switching either reloads once and then reuses.
    """
    return load_model(kind, ckpt_path)


def _source_name(obj) -> str:
    """Get a displayable name from either a Path or an UploadedFile."""
    if hasattr(obj, "name"):
        return str(obj.name)
    return str(Path(obj).name)


# -------------------------------------------------------------------
# STREAMLIT APP
# -------------------------------------------------------------------


def main():
    st.title("Polyp Detection – Patient Review Demo")

    # ----------------- MODEL SELECTION (SIDEBAR) -----------------

    family_label_to_kind: Dict[str, ModelKind] = {
        "Faster R-CNN": "fasterrcnn",
        "YOLO": "yolo",
    }

    family_label = st.sidebar.selectbox(
        "Model Architecture",
        options=list(family_label_to_kind.keys()),
    )
    model_kind: ModelKind = family_label_to_kind[family_label]

    # 2) Choose checkpoint for that family
    ckpt_files = list_checkpoints(model_kind)
    if not ckpt_files:
        st.error(
            f"No checkpoints found for `{model_kind}` in `{MODEL_DIRS[model_kind]}`.\n\n"
            "Place your trained models there, e.g.:\n"
            "- Faster R-CNN: models/faster_rcnn/best.pth\n"
            "- YOLO: models/yolo11s_best_det.pt"
        )
        return

    # Default selection: use best.* if present, else last in sorted list
    default_idx = 0
    for i, p in enumerate(ckpt_files):
        if p.name.lower().startswith("best"):
            default_idx = i
            break
    else:
        default_idx = len(ckpt_files) - 1

    selected_ckpt = st.sidebar.selectbox(
        "Model checkpoint",
        options=ckpt_files,
        index=default_idx,
        format_func=lambda p: p.name,
    )

    st.sidebar.caption(f"Using {family_label} checkpoint: `{selected_ckpt.name}`")

    # 3) Confidence threshold
    score_thr = st.sidebar.slider(
        "Confidence threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
    )
    st.sidebar.write("This threshold adjusts the confidence of each prediction.")
    st.sidebar.markdown(
        """
**Threshold note**

Only detections with a model score **≥ this value** are shown.

- Lower it to see **more** boxes (higher sensitivity, more false positives)
- Raise it to see **fewer** boxes (higher precision, more missed polyps)
"""
    )

    # ----------------- IMAGE SOURCE + INPUT -----------------

    st.header("Patient frames")

    source_choice = st.radio(
        "Image source",
        options=["Use built-in demo images", "Upload my own images"],
        index=0,
    )

    patient_id = st.text_input("Patient ID (optional)", value="")

    image_sources = []

    if source_choice == "Upload my own images":
        uploaded_files = st.file_uploader(
            "Upload all colonoscopy frames for this patient.",
            type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
            accept_multiple_files=True,
        )

        if not uploaded_files:
            st.info("Upload one or more images to get started.")
            return

        image_sources = list(uploaded_files)

        def get_image(i):
            f = image_sources[i]
            return f.name, Image.open(f).convert("RGB")

    else:  # Use built-in demo images
        demo_paths = _glob_images(DEMO_IMAGES_DIR)
        if not demo_paths:
            st.error(
                f"No demo images found in `{DEMO_IMAGES_DIR}`.\n"
                "Add a few .jpg/.png frames there or switch to upload mode."
            )
            return

        image_sources = demo_paths

        def get_image(i):
            p = image_sources[i]
            return p.name, Image.open(p).convert("RGB")

    num_images = len(image_sources)

    # ---- Determine series composition ----
    seq_files = [s for s in image_sources if "seq" in _source_name(s).lower()]
    single_files = [s for s in image_sources if "seq" not in _source_name(s).lower()]

    if seq_files and single_files:
        series_type = (
            f"Mixed: {len(seq_files)} sequence frame(s) "
            f"and {len(single_files)} single frame(s)"
        )
    elif seq_files:
        series_type = f"{len(seq_files)} sequence frame(s)."
    else:
        series_type = f"{len(single_files)} single-frame image(s)."

    st.markdown(
        f"**Patient summary:** {num_images} image(s) available. "
        f"Series type: `{series_type}`"
    )

    # Frame selection
    idx = st.selectbox(
        "Select frame to review",
        options=list(range(num_images)),
        format_func=lambda i: f"{i + 1}/{num_images} – {_source_name(image_sources[i])}",
    )

    img_name, image_pil = get_image(idx)

    # ----------------- RUN MODEL -----------------

    model = get_model(model_kind, str(selected_ckpt))
    preds = run_inference(model_kind, model, image_pil, score_thr=score_thr)

    # ----------------- DISPLAY IMAGES -----------------

    st.subheader("Original Image and Model Predictions")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Original Image**")
        st.image(
            image_pil,
            caption=f"{img_name} (frame {idx + 1}/{num_images})",
            use_container_width=True,
        )

    with col2:
        st.markdown("**Predictions (red boxes)**")
        img_with_boxes = draw_boxes(image_pil, preds)
        st.image(
            img_with_boxes,
            caption=f"Model predictions – {family_label}",
            use_container_width=True,
        )

    st.write(f"Found {len(preds)} detections with score ≥ {score_thr:.2f}.")

    # ----------------- DETECTIONS TABLE -----------------

    if preds:
        st.subheader("Detections summary")
        df = pd.DataFrame(
            [
                {
                    "Polyp ID": p["id"],
                    "Probability of true polyp": round(p["score"], 3),
                }
                for p in preds
            ]
        )
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No detections above the threshold.")

    # ----------------- DOCTOR REVIEW / FEEDBACK -----------------

    st.subheader("Doctor review")
    feedback_entries: List[dict] = []

    if preds:
        for p in preds:
            det_id = p["id"]
            score = p["score"]
            box = p["box"]

            with st.expander(f"Detection #{det_id} (score={score:.2f})", expanded=True):
                st.write(
                    f"Box: x1={int(box[0])}, y1={int(box[1])}, "
                    f"x2={int(box[2])}, y2={int(box[3])}"
                )

                marked_incorrect = st.checkbox(
                    "Mark this detection as incorrect (false positive)",
                    key=f"incorrect_{idx}_{det_id}",
                    value=False,
                )

                note = st.text_area(
                    "Optional note (e.g. 'fold', 'bubble', 'normal mucosa')",
                    key=f"note_{idx}_{det_id}",
                    height=60,
                )

                feedback_entries.append(
                    {
                        "det_id": det_id,
                        "box": box,
                        "score": score,
                        "marked_incorrect": marked_incorrect,
                        "note": note.strip(),
                    }
                )
    else:
        st.info("No detections above the threshold to review.")

    # Missed polyp feedback
    st.subheader("Missed polyps")
    missed_note = st.text_area(
        "If the model missed a polyp in this frame, describe where it is relative to the others.",
        key=f"missed_note_{idx}",
        height=80,
    )

    # ----------------- SAVE FEEDBACK -----------------

    if st.button("Save feedback for this frame"):
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "patient_id": patient_id,
            "num_images_for_patient": num_images,
            "frame_index": idx,
            "image_name": img_name,
            "score_threshold": score_thr,
            "model_family": model_kind,
            "model_checkpoint": selected_ckpt.name,
            "feedback_per_detection": feedback_entries,
            "missed_polyps_note": missed_note.strip(),
            "image_source": source_choice,
        }
        save_feedback(record)
        st.success("Feedback saved for this frame.")
        st.write(f"Appended to {FEEDBACK_FILE}")


if __name__ == "__main__":
    main()
