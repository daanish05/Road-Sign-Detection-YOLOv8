from ultralytics import YOLO

MODEL = "runs/detect/runs/road_sign_yolov8s/weights/best.pt"
SOURCE = "dataset/valid/images"

print("Loading YOLOv8s model...")

model = YOLO(MODEL)

print("\nRunning predictions on validation images...\n")

results = model.predict(
    source=SOURCE,
    imgsz=640,
    conf=0.10,
    iou=0.7,
    device=0,

    save=True,
    save_txt=True,
    save_conf=True,

    project="runs",
    name="yolov8s_validation_predictions",

    workers=0,
)

print("\n" + "=" * 60)
print("YOLOv8s VALIDATION PREDICTIONS COMPLETE")
print("=" * 60)

print(
    "Results saved to: "
    "runs/detect/runs/yolov8s_validation_predictions"
)