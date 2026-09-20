from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/runs/road_sign_yolov8s/weights/best.pt"
IMAGE_DIR = Path("dataset/valid/images")
LABEL_DIR = Path("dataset/valid/labels")

IMAGE_SIZE = 640
CONFIDENCE = 0.01
IOU_THRESHOLD = 0.50

# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("YOLOv8s VALIDATION ERROR ANALYSIS")
print("=" * 70)

print("\nLoading model...")
model = YOLO(MODEL_PATH)

# ============================================================
# RUN PREDICTIONS
# ============================================================

print("\nRunning predictions...")
results = model.predict(
    source=str(IMAGE_DIR),
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=0.7,
    device=0,
    workers=0,
    verbose=False,
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_iou(box1, box2):
    """
    Calculate IoU between two boxes.
    Boxes are [x1, y1, x2, y2].
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = intersection_width * intersection_height

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    if union <= 0:
        return 0.0

    return intersection / union


def load_ground_truth(label_path, image_width, image_height):
    """
    Read YOLO-format labels and convert them to
    [x1, y1, x2, y2].
    """

    ground_truth = []

    if not label_path.exists():
        return ground_truth

    with open(label_path, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        x_center = float(parts[1]) * image_width
        y_center = float(parts[2]) * image_height
        width = float(parts[3]) * image_width
        height = float(parts[4]) * image_height

        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2

        ground_truth.append({
            "class_id": class_id,
            "box": [x1, y1, x2, y2],
        })

    return ground_truth


# ============================================================
# ERROR COUNTERS
# ============================================================

total_ground_truth = 0
total_predictions = 0

correct_detections = 0
wrong_class = 0
missed = 0
false_positive = 0

matched_ious = []

class_stats = defaultdict(lambda: {
    "ground_truth": 0,
    "correct": 0,
    "wrong_class": 0,
    "missed": 0,
})

error_images = []

# ============================================================
# IMAGE-BY-IMAGE ANALYSIS
# ============================================================

for result in results:

    image_path = Path(result.path)
    image_name = image_path.name

    image_width = result.orig_shape[1]
    image_height = result.orig_shape[0]

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    gt = load_ground_truth(
        label_path,
        image_width,
        image_height
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions = []

    if result.boxes is not None and len(result.boxes) > 0:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            predictions.append({
                "class_id": int(cls),
                "box": box.tolist(),
                "confidence": float(conf),
            })

    total_ground_truth += len(gt)
    total_predictions += len(predictions)

    # --------------------------------------------------------
    # Track matches
    # --------------------------------------------------------

    matched_gt = set()
    matched_predictions = set()

    # Find best IoU prediction for every ground-truth object
    candidates = []

    for gt_index, gt_obj in enumerate(gt):

        for pred_index, pred_obj in enumerate(predictions):

            iou = calculate_iou(
                gt_obj["box"],
                pred_obj["box"]
            )

            if iou >= IOU_THRESHOLD:
                candidates.append(
                    (iou, gt_index, pred_index)
                )

    # Highest IoU first
    candidates.sort(reverse=True)

    for iou, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        gt_class = gt[gt_index]["class_id"]
        pred_class = predictions[pred_index]["class_id"]

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        class_stats[gt_class]["ground_truth"] += 1

        if gt_class == pred_class:

            correct_detections += 1
            class_stats[gt_class]["correct"] += 1
            matched_ious.append(iou)

        else:

            wrong_class += 1
            class_stats[gt_class]["wrong_class"] += 1

    # --------------------------------------------------------
    # Missed ground-truth objects
    # --------------------------------------------------------

    for gt_index, gt_obj in enumerate(gt):

        if gt_index not in matched_gt:

            missed += 1

            class_id = gt_obj["class_id"]

            class_stats[class_id]["ground_truth"] += 1
            class_stats[class_id]["missed"] += 1

    # --------------------------------------------------------
    # False positives
    # --------------------------------------------------------

    image_false_positive = 0

    for pred_index in range(len(predictions)):

        if pred_index not in matched_predictions:

            false_positive += 1
            image_false_positive += 1

    # --------------------------------------------------------
    # Save images containing errors
    # --------------------------------------------------------

    image_errors = (
        image_false_positive > 0
        or any(
            gt_index not in matched_gt
            for gt_index in range(len(gt))
        )
    )

    if image_errors:

        error_images.append({
            "image": image_name,
            "ground_truth": len(gt),
            "predictions": len(predictions),
            "false_positives": image_false_positive,
            "missed": sum(
                gt_index not in matched_gt
                for gt_index in range(len(gt))
            ),
        })


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("OVERALL ERROR ANALYSIS")
print("=" * 70)

print(f"\nConfidence threshold : {CONFIDENCE}")
print(f"IoU threshold        : {IOU_THRESHOLD}")

print(f"\nGround-truth objects : {total_ground_truth}")
print(f"Predicted objects    : {total_predictions}")

print("\n----------------------------------------")

print(f"Correct detections   : {correct_detections}")
print(f"Wrong-class matches  : {wrong_class}")
print(f"Missed objects       : {missed}")
print(f"False positives      : {false_positive}")

print("\n----------------------------------------")

if total_ground_truth > 0:

    detection_rate = (
        correct_detections / total_ground_truth
    )

    miss_rate = (
        missed / total_ground_truth
    )

    print(f"Correct detection rate : {detection_rate:.2%}")
    print(f"Miss rate              : {miss_rate:.2%}")

if matched_ious:

    print(
        f"Mean IoU of correct detections : "
        f"{np.mean(matched_ious):.4f}"
    )


# ============================================================
# PER-CLASS ERROR ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ERROR ANALYSIS")
print("=" * 70)

print(
    f"\n{'Class':>6} "
    f"{'GT':>6} "
    f"{'Correct':>10} "
    f"{'Wrong':>8} "
    f"{'Missed':>8}"
)

print("-" * 50)

for class_id in sorted(class_stats.keys()):

    stats = class_stats[class_id]

    print(
        f"{class_id:>6} "
        f"{stats['ground_truth']:>6} "
        f"{stats['correct']:>10} "
        f"{stats['wrong_class']:>8} "
        f"{stats['missed']:>8}"
    )


# ============================================================
# MOST PROBLEMATIC IMAGES
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION IMAGES WITH ERRORS")
print("=" * 70)

for item in error_images:

    print(
        f"\n{item['image']}"
        f"\n  Ground truth : {item['ground_truth']}"
        f"\n  Predictions  : {item['predictions']}"
        f"\n  False pos.   : {item['false_positives']}"
        f"\n  Missed       : {item['missed']}"
    )

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETE")
print("=" * 70)