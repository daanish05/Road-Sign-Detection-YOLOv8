from ultralytics import YOLO

MODEL = "runs/detect/runs/road_sign_yolov8s/weights/best.pt"
DATA = "dataset/data.yaml"

model = YOLO(MODEL)

thresholds = [0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50]

print("=" * 70)
print("YOLOv8s CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 70)

for conf in thresholds:
    print(f"\nConfidence threshold: {conf}")

    results = model.val(
        data=DATA,
        split="val",
        imgsz=640,
        batch=8,
        conf=conf,
        iou=0.7,
        device=0,
        workers=0,
        plots=False,
        verbose=False,
    )

    print(f"Precision : {results.box.mp:.4f}")
    print(f"Recall    : {results.box.mr:.4f}")
    print(f"mAP50     : {results.box.map50:.4f}")
    print(f"mAP50-95  : {results.box.map:.4f}")

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)