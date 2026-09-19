from ultralytics import YOLO

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL = "runs/detect/runs/road_sign_yolov8s/weights/best.pt"
DATA = "dataset/data.yaml"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading trained YOLOv8s model...")

model = YOLO(MODEL)

# --------------------------------------------------
# TEST EVALUATION
# --------------------------------------------------

print("\nRunning YOLOv8s evaluation on TEST set...\n")

results = model.val(
    data=DATA,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    project="runs",
    name="yolov8s_test_evaluation",
)

# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("\n" + "=" * 60)
print("YOLOv8s TEST RESULTS")
print("=" * 60)

print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")

print("=" * 60)