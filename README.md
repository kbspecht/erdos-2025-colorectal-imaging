# Image-Based Colorectal Polyp Detection (Erdös Institute Deep Learning Project Fall 2025)

Our main objective is to build a deep-learning system that detects colorectal polyps in images from colonoscopy. This will minimize missed lesions and improve patient outcomes by enabling more polyps that could be linked to colorectal cancer to be removed through colonoscopy.

The project uses high-quality public datasets [PolypGen](https://www.synapse.org/Synapse:syn26376615/wiki/613312), [REAL Colon](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866), and [Kvasir](https://datasets.simula.no/kvasir) (See [here](#datasets) for details).

We explored two models including Faster-RCNN and YOLO (v8,v11). A detailed model architecture description can be found for [YOLOv8](https://github.com/ultralytics/ultralytics/issues/189) and [Faster-RCNN](https://docs.pytorch.org/vision/main/_modules/torchvision/models/detection/faster_rcnn.html).

⚠️ Notebooks may contain medical images

## Team Members

[Betul Senay Aras](https://www.linkedin.com/in/betul-senay-aras-84318b70/), [Rebekah Eichberg](https://www.linkedin.com/in/rebekah-eichberg/), [Ruibo Zhang](https://www.linkedin.com/in/ruibo-zhang-b901161a1/), [Arthur Diep-Nguyen](https://www.linkedin.com/in/arthur-diep-nguyen/), [Kevin Specht](https://www.linkedin.com/in/kevin-specht-83aa4aab/)

## Deployment

Our models are deployed to a **web interface** using **Docker** and **Google Cloud Platform (GCP)**. The web app enables **doctors and medical staff** to upload patient results and perform **real-time inference** on medical images. Medical professionals can provide feedback regarding the model’s **accuracy and observations**.

All feedback is automatically saved to a structured **JSON file** for further analysis and model improvement.

**Live App:**
[Access the Application](https://polyp-app-50611727111.europe-west1.run.app/)
Please note that if you are using Safari, you may need to open a private window due to security restrictions and caches.

## File Organization

This repository is divided into six folders:
```bash
erdos-2025-colorectal-imaging/
├── deploy/                                           # implementation of web deployment
├── models/                                          # collection of network weights for trained models
│   ├── frcnn_imgsz640f2.pth
│   ├── frcnn_imgsz832f0.pth             # The weights for our best fast-rcnn model
│   └── ...
├── ⭐ notebooks/                            # collection of Jupyter notebooks
│   ├── EDA.ipynb                             # check this file for an overview of our dataset
│   ├── yolo_segmentation.ipynb               # train the best yolo model with segmentaion masks
│   └── ...
├── src/                      # model implementations
├── env.txt                # a minimum python environment for runnning all the scripts.
└── README.md
```

## Datasets

[PolypGen](https://www.synapse.org/Synapse:syn26376615/wiki/613312) 
This is a polyp segmentation and detection generalization dataset composed of 8037 frames including both single and sequence frames. This dataset contains 3762 positive sample frames (frames containing annotated  polyps) and 4275 negative frames. These data are collected from six different hospitals.

[REAL Colon](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866) 
This is a dataset comprising 60 recordings of real-world colonoscopies which come from four different clinical studies with each study contributing 15 videos.

[Kvasir](https://datasets.simula.no/kvasir) 
This is a dataset consisting of 1000 images, annotated and verified by endoscopists, to be used for different tasks such as deep learning and transfer learning.

## Stakeholders

- Endoscopists looking to improve accuracy, speed, and clinical workflow integration
- Pathologists validating findings in imaging and support diagnoses
- Hospital management looking to reduce procedure times, lower costs and improve patient outcomes while not interfering with accuracy
- Medtech developers providing commercialization and integration with existing endoscopy hardware and practices

## Key Performance Indicators (KPI)

- Intersection over Union (IoU) measuring how well predicted bounding box for polyp detection overlaps with ground truth box
- Precision measuring proportion of true positives (predictions where IoU exceeds given threshold) among all predicted positives
- Recall measuring proportion of true positives among all actual positives
- Mean Average Precision (mAP) measuring mean of average precisions (area under precision-recall curve) across multiple IoU thresholds from 0.5 to 0.95 in increments of 0.05
