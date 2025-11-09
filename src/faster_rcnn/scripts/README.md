
# Faster R-CNN Polyp Detection Pipeline

This repository provides a **reproducible end-to-end training and evaluation pipeline** for colorectal polyp detection using Faster R-CNN.

It replaces the previous Jupyter notebook workflow and is designed for **local runs**.  

---

## A Small Note:

Depending on user preference, this pipeline can be run from CLI or from [this notebook](https://github.com/kbspecht/erdos-2025-colorectal-imaging/edit/main/src/faster_rcnn/scripts/README.md#:~:text=run_faster_rcnn.ipynb).

---

## Directory Structure

```
project-root/
│
├── models/
│   └── faster_rcnn/              # Final production-ready checkpoints (.pth)
│
├── runs/
│   └── frcnn_polyp/              # Temporary per-run checkpoints/logs
│
├── src/
│   └── faster_rcnn/
│       ├── data/                 # Data prep + loaders
│       ├── training/             # Training scripts
│       └── scripts/
│           └── train_eval_faster_rcnn.py
│
└── testing_faster/               # Example dataset root
```

---

## 1. Environment Setup

```bash
# Create and activate environment
conda create -n erdos_fall_2025 python=3.10 -y
conda activate erdos_fall_2025

# Install requirements
pip install -r requirements.txt
```

---

## 2. Run the End-to-End Pipeline

```bash
cd /path/to/erdos-2025-colorectal-imaging
export PYTHONPATH=src:$PYTHONPATH

python src/faster_rcnn/scripts/train_eval_faster_rcnn.py     --datasets testing_faster     --images-root testing_faster/images/val     --epochs 3
```

This command will:
1. Prepare COCO-style dataset artifacts (`train.json`, `val.json`, `roots_map.json`)
2. Train the Faster R-CNN model for the specified number of epochs
3. Save checkpoints in `models/faster_rcnn/` (with unique filenames)
4. Run COCO mAP evaluation and print metrics to the console

---

## 3. Skip Data Preparation

If artifacts already exist (to save time):

```bash
python src/faster_rcnn/scripts/train_eval_faster_rcnn.py     --datasets testing_faster     --images-root testing_faster/images/val     --epochs 3     --skip-prepare
```

---

## 4. Full Argument Reference

### Core Arguments

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--datasets` | `Path` | `testing_faster` | Root directory for raw datasets. |
| `--images-root` | `Path` | *(required)* | Directory containing images for training/validation. |
| `--artifacts-dir` | `Path` | `src/artifacts` | Output directory for `train.json`, `val.json`, `roots_map.json`. |
| `--epochs` | `int` | `1` | Number of training epochs. |
| `--skip-prepare` | flag | `False` | Skip data preparation if artifacts already exist. |

### Training Hyperparameters

All hyperparameters are configurable directly via CLI flags:

| Argument | Type | Default | Description |
|-----------|------|----------|-------------|
| `--batch-size` | `int` | `8` | Training batch size. |
| `--num-workers` | `int` | `0` | Number of data loader workers. |
| `--img-size` | `int` | `640` | Image resize dimension. |
| `--train-augs` | `str` | `medium` | Augmentation preset (`light`, `medium`, `strong`). |
| `--num-classes` | `int` | `2` | Number of target classes (including background). |
| `--freeze-backbone` | `int` | `2` | Number of backbone layers to freeze. |
| `--opt` | `str` | `sgd` | Optimizer type (`sgd`, `adamw`, etc.). |
| `--lr-backbone` | `float` | `1e-4` | Learning rate for the backbone. |
| `--lr-heads` | `float` | `5e-3` | Learning rate for detection heads. |
| `--weight-decay` | `float` | `1e-4` | Weight decay for regularization. |
| `--lr-scheduler` | `str` | `none` | Learning rate scheduler (`cosine`, `step`, or `none`). |
| `--device` | `str` | `cpu` | Device used for training (`cpu`, `cuda`, `mps`). |

---

## Example Command with Explicit Hyperparameters

```bash
python src/faster_rcnn/scripts/train_eval_faster_rcnn.py     --datasets testing_faster     --images-root testing_faster/images/val     --epochs 5     --batch-size 4     --lr-heads 0.002     --lr-backbone 0.0001     --opt adamw     --train-augs strong     --device cuda
```

---

## 5. Evaluate a Specific Checkpoint

To evaluate a trained checkpoint (e.g. `models/faster_rcnn/frcnn_imgsz832f0_best.pth`):

```bash
python src/faster_rcnn/scripts/train_eval_faster_rcnn.py     --datasets testing_faster     --images-root testing_faster/images/val     --epochs 0     --skip-prepare
```

This will skip training and just run COCO-style evaluation.

---

## 🧰 6. Outputs and Logging

After each run, the following are generated:

- **Data Artifacts:**  
  `src/artifacts/train.json`, `val.json`, `roots_map.json`
- **Model Checkpoints:**  
  `models/faster_rcnn/epoch_001_<timestamp>.pth`
- **Evaluation Metrics:**  
  Printed in terminal:
  ```
  Final metrics:
    mAP@[.5:.95] = 0.4271
    mAP@0.5      = 0.6938
    Recall       = 0.8023
    Precision    = 0.7442
  ```

---

## Notes

- The script automatically avoids overwriting existing checkpoints by appending unique suffixes.
- Compatible with both macOS (MPS) and Linux (CUDA or CPU).

