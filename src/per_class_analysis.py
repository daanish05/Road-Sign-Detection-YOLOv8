from ultralytics import YOLO
import numpy as np

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

MODEL = "runs/detect/runs/road_sign_baseline-4/weights/best.pt"
DATA = "dataset/data.yaml"

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

print("Loading trained YOLOv8 model...")
model = YOLO(MODEL)

# --------------------------------------------------
# VALIDATE
# --------------------------------------------------

print("\nRunning validation on validation set...\n")

results = model.val(
    data=DATA,
    split="val",
    imgsz=640,
    batch=8,
    device=0,
    workers=0,
    plots=True,
    verbose=True
)

# --------------------------------------------------
# OVERALL METRICS
# --------------------------------------------------

print("\n" + "=" * 70)
print("OVERALL VALIDATION RESULTS")
print("=" * 70)

print(f"Precision : {results.box.mp:.4f}")
print(f"Recall    : {results.box.mr:.4f}")
print(f"mAP50     : {results.box.map50:.4f}")
print(f"mAP50-95  : {results.box.map:.4f}")

# --------------------------------------------------
# PER-CLASS METRICS
# --------------------------------------------------

print("\n" + "=" * 70)
print("PER-CLASS RESULTS AVAILABLE FROM VALIDATION")
print("=" * 70)

class_names = results.names

precision = np.asarray(results.box.p)
recall = np.asarray(results.box.r)
map50 = np.asarray(results.box.ap50)
map50_95 = np.asarray(results.box.ap)

# Ultralytics stores only classes with validation metrics.
# The class IDs are available through results.box.ap_class_index.

class_ids = np.asarray(results.box.ap_class_index)

print(f"\nClasses with validation metrics: {len(class_ids)}")
print(f"Total dataset classes: {len(class_names)}\n")

print(
    f"{'ID':>4} | "
    f"{'Class Name':<30} | "
    f"{'P':>7} | "
    f"{'R':>7} | "
    f"{'mAP50':>7} | "
    f"{'mAP50-95':>9}"
)

print("-" * 80)

for i, class_id in enumerate(class_ids):

    class_id = int(class_id)

    name = class_names[class_id]

    print(
        f"{class_id:4d} | "
        f"{name:<30} | "
        f"{precision[i]:7.4f} | "
        f"{recall[i]:7.4f} | "
        f"{map50[i]:7.4f} | "
        f"{map50_95[i]:9.4f}"
    )

# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"Dataset classes              : {len(class_names)}"
)

print(
    f"Classes represented in val   : {len(class_ids)}"
)

print(
    f"Classes without val instances: "
    f"{len(class_names) - len(class_ids)}"
)

print("\nAnalysis complete.")