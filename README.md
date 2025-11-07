# erdos-2025-colorectal-imaging

## Image-Based Colorectal Polyp Detection (Erdös Institute Deep Learning Project Fall 2025)

Betul Senay Aras, Rebekah Eichberg, Ruibo Zhang, Arthur Diep-Nguyen, Kevin Specht

### Objectives

Our main objective is to build a deep-learning system that detects colorectal polyps in images from colonoscopy. This will minimize missed lesions and improve patient outcomes by enabling more polyps that could be linked to colorectal cancer to be removed through colonoscopy.

### Datasets

PolypGen: https://www.synapse.org/Synapse:syn26376615/wiki/613312 \
This is a polyp segmentation and detection generalization dataset composed of 8037 frames including both single and sequence frames. This dataset contains 3762 positive sample frames (frames containing annotated  polyps) and 4275 negative frames. These data are collected from six different hospitals.

Real Colon: https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866 \
This is a dataset comprising 60 recordings of real-world colonoscopies which come from four different clinical studies with each study contributing 15 videos.

Kvasir: https://datasets.simula.no/kvasir \
This is a dataset consisting of 1000 images, annotated and verified by endoscopists, to be used for different tasks such as deep learning and transfer learning.

### Stakeholders

- Endoscopists looking to improve accuracy, speed, and clinical workflow integration
- Pathologists validating findings in imaging and support diagnoses
- Hospital management looking to reduce procedure times, lower costs and improve patient outcomes while not interfering with accuracy
- Medtech developers providing commercialization and integration with existing endoscopy hardware and practices

### Key Performance Indicators (KPI)
- Intersection over Union (IoU) measuring how well predicted bounding box for polyp detection overlaps with ground truth box
- Precision measuring proportion of true positives (predictions where IoU exceeds given threshold) among all predicted positives
- Recall measuring proportion of true positives among all actual positives
- Mean Average Precision (mAP) measuring mean of average precisions (area under precision-recall curve) across multiple IoU thresholds from 0.5 to 0.95 in increments of 0.05

### File Organization
This repository is divided into six folders:

#### Check-In
This folder contains the check-in documents for the project.

#### architectures
This folder contains various architectures for deploying the models.

#### configs
This folder contains various configuration files for training and running the models.

#### notebooks
This folder contains various notebooks for training and running the models.

#### runs
This folder contains the results of various runs of the models.

#### src
This folder contains various scripts for training and running the models.
