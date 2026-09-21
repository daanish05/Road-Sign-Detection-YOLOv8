# Road Sign Detection Using YOLOv8

An AI-based road sign detection system developed using YOLOv8 and a custom traffic-sign dataset containing 264 classes.

The project performs object detection on road-sign images and provides:

- Road-sign localization using bounding boxes
- Traffic-sign class prediction
- Prediction confidence scores
- Annotated output images
- Interactive web-based inference using Flask

---

## 1. Project Overview

Road signs play an important role in road safety and intelligent transportation systems.

This project explores the use of deep learning and object detection to automatically identify road signs from images.

The system uses **YOLOv8s (You Only Look Once)** for object detection and provides a Flask-based web interface through which users can upload an image and obtain detection results.

The project includes dataset analysis, data validation, source-group-aware data splitting, model training, evaluation, error analysis, and deployment through a local web application.

---

## 2. Features

### Machine Learning

- YOLOv8-based object detection
- 264 traffic-sign classes
- GPU-accelerated inference
- Bounding-box detection
- Confidence scores
- Custom trained model

### Web Application

- Upload road-sign images
- Automatic image processing
- Detection results displayed in the browser
- Annotated output image
- Predicted class names
- Class IDs
- Confidence scores
- Processing time
- Analyze multiple images without restarting the application

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Object Detection | Ultralytics YOLOv8 |
| Web Framework | Flask |
| Image Processing | OpenCV |
| Image Handling | Pillow |
| Dataset Format | YOLO |
| GPU | NVIDIA CUDA |
| Frontend | HTML, CSS, JavaScript |

---

## 4. Model

The final application uses:

**Model:** YOLOv8s

**Input Size:** 640 × 640

**Number of Classes:** 264

**Training Dataset:** Stratified train/validation split

**Checkpoint:**

```text
runs/detect/runs/road_sign_yolov8s_stratified/weights/best.pt