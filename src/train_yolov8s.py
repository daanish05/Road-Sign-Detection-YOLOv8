from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL = "yolov8s.pt"
DATA = "dataset/data.yaml"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8

PROJECT = "runs"
RUN_NAME = "road_sign_yolov8s"

# --------------------------------------------------
# LOAD PRETRAINED MODEL
# --------------------------------------------------

print("Loading YOLOv8s pretrained model...")

model = YOLO(MODEL)

# --------------------------------------------------
# TRAIN
# --------------------------------------------------

print("\nStarting YOLOv8s training...\n")

results = model.train(
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

    # Required for your Windows/Python setup
    workers=0,
)

print("\n" + "=" * 60)
print("YOLOv8s TRAINING COMPLETE")
print("=" * 60)

print(f"Run directory: {PROJECT}/detect/{RUN_NAME}")