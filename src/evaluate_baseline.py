from ultralytics import YOLO


MODEL = "runs/detect/runs/road_sign_baseline/weights/best.pt"
DATA = "dataset/data.yaml"


print("=" * 70)
print("YOLOv8 BASELINE - TEST SET EVALUATION")
print("=" * 70)

model = YOLO(MODEL)

results = model.val(
    data=DATA,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    plots=True,
)

print("\n" + "=" * 70)
print("TEST EVALUATION COMPLETE")
print("=" * 70)

print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")
print(f"mAP50:     {results.box.map50:.4f}")
print(f"mAP50-95:  {results.box.map:.4f}")