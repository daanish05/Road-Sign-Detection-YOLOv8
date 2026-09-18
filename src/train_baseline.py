from ultralytics import YOLO


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL = "yolov8n.pt"
DATA = "dataset/data.yaml"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 8 # Using GPU

PROJECT = "runs"
RUN_NAME = "road_sign_baseline"


# ---------------------------------------------------------
# Load pretrained YOLOv8 model
# ---------------------------------------------------------

print("=" * 70)
print("YOLOv8 ROAD SIGN DETECTION - BASELINE TRAINING")
print("=" * 70)

print("\nLoading model...")
model = YOLO(MODEL)

print(f"Model: {MODEL}")
print(f"Dataset: {DATA}")
print(f"Epochs: {EPOCHS}")
print(f"Image size: {IMAGE_SIZE}")
print(f"Batch size: {BATCH_SIZE}")


# ---------------------------------------------------------
# Train
# ---------------------------------------------------------

results = model.train(
    data=DATA,

    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,

    # Training control
    patience=20,

    # Use pretrained weights
    pretrained=True,

    # Save checkpoints
    save=True,
    save_period=10,

    # Generate plots
    plots=True,

    # Output directory
    project=PROJECT,
    name=RUN_NAME,

    # Reproducibility
    seed=42,

    # Automatically use GPU if available
    device=0, #previously "0" for GPU,(Using GPU)
    workers=0, 
)


print("\n" + "=" * 70)
print("BASELINE TRAINING COMPLETE")
print("=" * 70)

print(f"\nResults saved in: {PROJECT}/detect/{RUN_NAME}")


