from pathlib import Path
import time

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/runs/road_sign_yolov8s_stratified/weights/best.pt"

# Change this to your input image
SOURCE = "test_image.jpeg"

IMAGE_SIZE = 640

# Confidence threshold
CONFIDENCE = 0.001

# IoU threshold for NMS
IOU = 0.45

# GPU
DEVICE = 0

# Output directory
PROJECT = "runs"
RUN_NAME = "final_predictions"


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading YOLOv8s model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print(f"Model: {MODEL_PATH}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Confidence threshold: {CONFIDENCE}")


# ============================================================
# RUN DETECTION
# ============================================================

print("\nRunning detection...")

start_time = time.perf_counter()

results = model.predict(
    source=SOURCE,
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=IOU,
    device=DEVICE,
    save=True,
    save_txt=True,
    save_conf=True,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=True,
    verbose=False,
)

end_time = time.perf_counter()

inference_time = (end_time - start_time) * 1000


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("ROAD SIGN DETECTION RESULTS")
print("========================================")

total_detections = 0

for result in results:

    if result.boxes is None or len(result.boxes) == 0:
        print("\nNo road signs detected.")
        continue

    boxes = result.boxes

    total_detections += len(boxes)

    print(f"\nDetected signs: {len(boxes)}")

    for i, box in enumerate(boxes):

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"Detection {i + 1}: "
            f"{class_name} "
            f"({confidence * 100:.2f}%)"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n========================================")
print("SUMMARY")
print("========================================")

print(f"Total detections : {total_detections}")
print(f"Inference time   : {inference_time:.2f} ms")

print(
    f"Output directory : "
    f"{PROJECT}/{RUN_NAME}"
)

print("========================================")