from ultralytics import YOLO

MODEL = "runs/detect/runs/road_sign_yolov8s_stratified/weights/best.pt"
SOURCE = "dataset/stratified/valid/images"

model = YOLO(MODEL)

results = model.predict(
    source=SOURCE,
    imgsz=640,
    conf=0.25,
    iou=0.5,
    device=0,
    workers=0,
    save=True,
    save_txt=True,
    save_conf=True,
    project="runs",
    name="stratified_valid_predictions",
)

print("\nPrediction completed.")
print("Results saved to:")
print("runs/detect/runs/stratified_valid_predictions")