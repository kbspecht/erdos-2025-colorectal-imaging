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
[Access the Application](https://polyp-app-50611727111.europe-west1.run.app)
Please note that if you are using Safari, you may need to open a private window due to security restrictions and caches.

---

## Using the Web App

Once the app is running locally (`http://localhost:8080`) or deployed (Cloud Run URL), you’ll see an interface that allows you to:

1. **Upload Images or Folders**
   - Upload a single frame, or an entire patient folder containing multiple frames.
   - The app automatically detects if an upload contains a sequence.
   - User can also select demo images so they don't need to upload their own images.

2. **Run Inference**
   - After selecting a model on the left, the results will show.
   - The results are displayed side-by-side:
     - Left: Original image
     - Right: Predicted bounding boxes (polyps highlighted)
   - Feel free to adjust the IoU threshold for desired predictions.

3. **Provide Feedback (Optional)**
   - Experts can mark predictions as correct or incorrect.
   - Feedback is stored in `feedback/feedback.jsonl` for future retraining.

4. **Switch Between Models**
   - Use the sidebar dropdown to select between model types (e.g., Faster R-CNN and YOLOv8).

---

### Interface Overview

| Section | Description |
|----------|--------------|
| **Sidebar** | Model selection, checkpoint path, IoU threshold, definitions |
| **Main Panel** | Upload interface and visual results |
| **Results Viewer** | Displays predicted bounding boxes and provide feedback |

---

## File Organization

This repository is divided into six folders:
```bash
erdos-2025-colorectal-imaging/
├── configs/
├── demo_images/                       # Images to be used in deployment demo
├── deploy/                            # implementation of web deployment
├── feedback/                          # Storage of the JSON file from web app
├── models/                            # Our pretrained models
│   ├── yolo                           # The weights for our best yolo models
│   ├── faster_rcnn                    # The weights for our best fast-rcnn model
├── notebooks/                         # collection of Jupyter notebooks
│   ├── EDA.ipynb                      # check this file for an overview of ourdataset
│   ├── yolo_segmentation.ipynb        # train the best yolo model with segmentaion masks
│   ├── run_faster_rcnn.ipynb          # run and evaluate the faster_rcnn model 
│   └── ...
├── runs/                              # Delete?
├── src/                               # model implementations
├── env.txt                            # a minimum python environment for runnning all the scripts
├── app.py                             # Script for Streamlit app
├── Dockerfile                            
├── README.md
└── requirements.txt                   # Necessary requirements for Dockerfile
```

## Datasets

[PolypGen](https://www.synapse.org/Synapse:syn26376615/wiki/613312) 
This is a polyp segmentation and detection generalization dataset composed of 8037 frames including both single and sequence frames. This dataset contains 3762 positive sample frames (frames containing annotated  polyps) and 4275 negative frames. These data are collected from six different hospitals. PolypGen has a total number of **1347** single images.

[REAL Colon](https://plus.figshare.com/articles/media/REAL-colon_dataset/22202866) 
This dataset comprises 60 recordings of real-world colonoscopies, which come from four different clinical studies, with each study contributing 15 videos. Each video records all the frames captured by a camera during a complete colonoscopy procedure. The total number of distinct polyps or distinct displays for the identical polyps is estimated to be around 500. 

[Kvasir](https://datasets.simula.no/kvasir) 
This is a dataset consisting of 1000 single images, annotated and verified by endoscopists, to be used for different tasks such as deep learning and transfer learning. The polyps in this dataset can be located in either the upper GI tract (esophagus, stomach) or the lower GI tract (colon, rectum).

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
  
## Results on Validation Set 
| Model  | Precision | Recall  | mAP@50 | mAP@50:95 |
| ------------- | ------------- |------------- | ------------- |------------- |
| YOLO 8s (Baseline)  | 0.83  | 0.84  | 0.82  | 0.64  |
| YOLO 11m-det  | 0.92  | 0.87  | 0.94  | 0.74  |
| YOLO 11s-seg Bounding Box | 0.92 | 0.89  | 0.94  | 0.77  |
| YOLO 11s-seg Mask  | 0.92  | 0.89  | 0.94  | 0.75  |

## Results on Test Set Single Images 
| Model  | Precision | Recall  | mAP@50 | mAP@50:95 |
| ------------- | ------------- |------------- | ------------- |------------- |
| YOLO 8s (Baseline)  | 0.88  | 0.78  | 0.86  | 0.65  |
| YOLO 11m-det  | 0.94  | 0.84  | 0.92  | 0.71  |
| YOLO 11s-seg Bounding Box | 0.94 | 0.84  | 0.92  | 0.71  |
| YOLO 11s-seg Mask  | 0.93  | 0.83  | 0.91  | 0.70  |

## How-To Run:
- YOLO models
- [Faster R-CNN](https://github.com/kbspecht/erdos-2025-colorectal-imaging/blob/main/notebooks/README.md)
