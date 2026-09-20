from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIG
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
print("YOLOv8s CLASS CONFUSION ANALYSIS")
print("=" * 70)

model = YOLO(MODEL_PATH)

print("\nRunning validation predictions...")

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
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    if union <= 0:
        return 0

    return intersection / union


# ============================================================
# LOAD GROUND TRUTH
# ============================================================

def load_labels(label_path, width, height):

    labels = []

    if not label_path.exists():
        return labels

    with open(label_path, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            xc = float(parts[1]) * width
            yc = float(parts[2]) * height
            w = float(parts[3]) * width
            h = float(parts[4]) * height

            x1 = xc - w / 2
            y1 = yc - h / 2
            x2 = xc + w / 2
            y2 = yc + h / 2

            labels.append({
                "class": class_id,
                "box": [x1, y1, x2, y2]
            })

    return labels


# ============================================================
# CONFUSION COUNTER
# ============================================================

confusion = defaultdict(int)

correct = 0
wrong_class = 0
missed = 0

# ============================================================
# PROCESS IMAGES
# ============================================================

for result in results:

    image_path = Path(result.path)

    width = result.orig_shape[1]
    height = result.orig_shape[0]

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    ground_truth = load_labels(
        label_path,
        width,
        height
    )

    predictions = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(
            boxes,
            classes,
            confidences
        ):

            predictions.append({
                "class": int(cls),
                "box": box.tolist(),
                "confidence": float(conf)
            })

    matched_gt = set()
    matched_pred = set()

    candidates = []

    # Find all sufficiently overlapping pairs
    for gt_index, gt in enumerate(ground_truth):

        for pred_index, pred in enumerate(predictions):

            iou = calculate_iou(
                gt["box"],
                pred["box"]
            )

            if iou >= IOU_THRESHOLD:

                candidates.append(
                    (
                        iou,
                        gt_index,
                        pred_index
                    )
                )

    # Highest IoU first
    candidates.sort(reverse=True)

    for iou, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_pred:
            continue

        gt_class = ground_truth[gt_index]["class"]
        pred_class = predictions[pred_index]["class"]

        matched_gt.add(gt_index)
        matched_pred.add(pred_index)

        if gt_class == pred_class:

            correct += 1

        else:

            wrong_class += 1

            confusion[
                (gt_class, pred_class)
            ] += 1

    # Missed objects
    for gt_index in range(len(ground_truth)):

        if gt_index not in matched_gt:

            missed += 1


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 70)
print("RESULT")
print("=" * 70)

print(f"\nCorrect detections : {correct}")
print(f"Wrong-class        : {wrong_class}")
print(f"Missed             : {missed}")

print("\n" + "=" * 70)
print("TOP CLASS CONFUSIONS")
print("=" * 70)

if not confusion:

    print("\nNo class confusions found.")

else:

    sorted_confusion = sorted(
        confusion.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(
        f"\n{'GT Class':>10} "
        f"{'Pred Class':>12} "
        f"{'Count':>8}"
    )

    print("-" * 35)

    for (gt_class, pred_class), count in sorted_confusion:

        print(
            f"{gt_class:>10} "
            f"{pred_class:>12} "
            f"{count:>8}"
        )


print("\n" + "=" * 70)
print("CLASS CONFUSION ANALYSIS COMPLETE")
print("=" * 70)