# Road Sign Detection Using YOLOv8

A computer vision project for detecting and classifying road traffic
signs using the YOLOv8 object detection framework.

## Dataset

The project uses the Traffic Sign Recognition YOLOv8 dataset
available on Kaggle:

https://www.kaggle.com/datasets/lara311/traffic-sign-recognition-yolov8

The dataset is not included in this repository.

### Dataset structure

After downloading and extracting the dataset, place it inside:

dataset/

The expected structure is:

dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/