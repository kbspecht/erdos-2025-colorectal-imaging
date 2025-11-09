How-To Run Faster-RCNN:

## Description of Each Notebook

| Notebook | Purpose |
|-----------|----------|
| **1. EDA.ipynb** | Performs Exploratory Data Analysis (EDA) on Polypgene data. Analyzes the distribution of single vs. sequence images, total image counts, bounding-box statistics, and dataset composition across centers and sources. |
| **2. data_cleaning.ipynb** | Cleans raw annotations by removing incorrect bounding boxes, thin or noisy boxes, and inconsistent sequence labels.|
| **3. train_val_test_split.ipynb** | Ensures data integrity and generalization by carefully splitting the dataset into train, validation, and test sets. Prevents data leakage by assigning entire sequences to a single split and reserves Center C6 exclusively for testing to evaluate model generalizability.|
| **4. yolo_detection_labeling.ipynb** | Scans dataset folders, applies static augmentations to single-center training images, and generates YOLO detection labels in a modular, folder-aware structure.|
| **5. yolo_detection_model_train.ipynb** | Fine-tunes the YOLO11s model for polyp detection and evaluates performance on the validation set, generating key plots such as precision–recall curves and confusion matrices. |
| **6. yolo_segmentation.ipynb** | Generates static augmentations for single training images, creates YOLO segmentation labels for multiple data folders, builds the final YOLO training split, and trains YOLO11s-Seg using built-in augmentations and multi-scale training. |
| **7. yolo_evaluate_models_test_set.ipynb** | Evaluates trained models (detection or segmentation) on test sets, generating Markdown summaries and performance plots automatically. |
| **8. false_analysis_yolo_models.ipynb** |  Analyzes false positives and false negatives from trained models to identify common error patterns and gain further insights for model improvement.|
