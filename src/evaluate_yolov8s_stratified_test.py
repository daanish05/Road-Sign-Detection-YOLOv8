from ultralytics import YOLO

MODEL = "runs/detect/runs/road_sign_yolov8s_stratified/weights/best.pt"
DATA = "dataset/data.yaml"

model = YOLO(MODEL)

results = model.val(
    data=DATA,
    split="test",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    save_json=True,
)

print("\n========================================")
print("YOLOv8s STRATIFIED 640 TEST RESULTS")
print("========================================")
print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")
print("========================================")