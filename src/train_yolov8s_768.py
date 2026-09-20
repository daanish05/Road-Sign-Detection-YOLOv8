from ultralytics import YOLO

# =========================
# Configuration
# =========================

MODEL = "yolov8s.pt"
DATA = "dataset/data_stratified.yaml"

EPOCHS = 100
IMAGE_SIZE = 768
BATCH_SIZE = 4

PROJECT = "runs"
RUN_NAME = "road_sign_yolov8s_768"

# =========================
# Load pretrained model
# =========================

model = YOLO(MODEL)

# =========================
# Train
# =========================

model.train(
    data=DATA,
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    patience=20,
    pretrained=True,
    save=True,
    save_period=10,
    plots=True,
    project=PROJECT,
    name=RUN_NAME,
    seed=42,
    device=0,
    workers=0,
)