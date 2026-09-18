from ultralytics import YOLO


MODEL = "runs/detect/runs/road_sign_baseline/weights/best.pt"
SOURCE = "dataset/test/images"

model = YOLO(MODEL)

results = model.predict(
    source=SOURCE,
    imgsz=640,
    conf=0.10,
    device=0,
    save=True,
    save_txt=True,
    save_conf=True,
    project="runs",
    name="baseline_test_predictions",
)

print("\nPrediction complete.")
print("Results saved to:")
print("runs/detect/baseline_test_predictions")