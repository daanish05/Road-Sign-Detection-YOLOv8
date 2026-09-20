from ultralytics import YOLO
from pathlib import Path
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "runs/detect/runs/road_sign_yolov8s/weights/best.pt"

TRAIN_LABEL_DIR = Path("dataset/train/labels")
VALID_IMAGE_DIR = Path("dataset/valid/images")
VALID_LABEL_DIR = Path("dataset/valid/labels")

IMAGE_SIZE = 640
CONFIDENCE = 0.01
IOU_THRESHOLD = 0.50


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 75)
print("CLASS CONFUSION + TRAINING FREQUENCY ANALYSIS")
print("=" * 75)

model = YOLO(MODEL_PATH)

names = model.names


# ============================================================
# COUNT TRAINING INSTANCES
# ============================================================

print("\nCounting training instances...")

train_counts = defaultdict(int)

for label_file in TRAIN_LABEL_DIR.glob("*.txt"):

    with open(label_file, "r") as f:

        for line in f:

            parts = line.strip().split()

            if len(parts) == 5:

                class_id = int(parts[0])

                train_counts[class_id] += 1


# ============================================================
# IOU FUNCTION
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
        return 0.0

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
# RUN VALIDATION PREDICTIONS
# ============================================================

print("\nRunning validation predictions...")

results = model.predict(
    source=str(VALID_IMAGE_DIR),
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    iou=0.7,
    device=0,
    workers=0,
    verbose=False,
)


# ============================================================
# CONFUSION COUNTER
# ============================================================

confusion = defaultdict(int)


# ============================================================
# MATCH PREDICTIONS TO GROUND TRUTH
# ============================================================

for result in results:

    image_path = Path(result.path)

    width = result.orig_shape[1]
    height = result.orig_shape[0]

    label_path = (
        VALID_LABEL_DIR /
        f"{image_path.stem}.txt"
    )

    ground_truth = load_labels(
        label_path,
        width,
        height
    )

    predictions = []

    if result.boxes is not None:

        boxes = result.boxes.xyxy.cpu().numpy()

        classes = (
            result.boxes.cls
            .cpu()
            .numpy()
            .astype(int)
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .numpy()
        )

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
    matched_predictions = set()

    candidates = []

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

    candidates.sort(reverse=True)

    for iou, gt_index, pred_index in candidates:

        if gt_index in matched_gt:
            continue

        if pred_index in matched_predictions:
            continue

        gt_class = ground_truth[gt_index]["class"]
        pred_class = predictions[pred_index]["class"]

        matched_gt.add(gt_index)
        matched_predictions.add(pred_index)

        if gt_class != pred_class:

            confusion[
                (gt_class, pred_class)
            ] += 1


# ============================================================
# OUTPUT
# ============================================================

print("\n" + "=" * 75)
print("TOP CLASS CONFUSIONS WITH TRAINING FREQUENCY")
print("=" * 75)

sorted_confusion = sorted(
    confusion.items(),
    key=lambda x: x[1],
    reverse=True
)

print()

print(
    f"{'GT':>5} "
    f"{'GT NAME':<30} "
    f"{'PRED':>5} "
    f"{'PRED NAME':<30} "
    f"{'GT TRAIN':>9} "
    f"{'PRED TRAIN':>10} "
    f"{'COUNT':>7}"
)

print("-" * 110)

for (gt_class, pred_class), count in sorted_confusion:

    gt_name = str(names.get(gt_class, "UNKNOWN"))
    pred_name = str(names.get(pred_class, "UNKNOWN"))

    gt_train = train_counts.get(gt_class, 0)
    pred_train = train_counts.get(pred_class, 0)

    print(
        f"{gt_class:>5} "
        f"{gt_name[:30]:<30} "
        f"{pred_class:>5} "
        f"{pred_name[:30]:<30} "
        f"{gt_train:>9} "
        f"{pred_train:>10} "
        f"{count:>7}"
    )


# ============================================================
# RARE CLASSES
# ============================================================

print("\n" + "=" * 75)
print("VALIDATION CONFUSIONS INVOLVING RARE TRAINING CLASSES")
print("=" * 75)

print("\nClasses with <= 5 training instances:")

for (gt_class, pred_class), count in sorted_confusion:

    gt_train = train_counts.get(gt_class, 0)
    pred_train = train_counts.get(pred_class, 0)

    if gt_train <= 5 or pred_train <= 5:

        gt_name = str(names.get(gt_class, "UNKNOWN"))
        pred_name = str(names.get(pred_class, "UNKNOWN"))

        print(
            f"\nGT   {gt_class} ({gt_name})"
            f"\n     training instances: {gt_train}"
            f"\n"
            f"Pred {pred_class} ({pred_name})"
            f"\n     training instances: {pred_train}"
            f"\nConfusion count: {count}"
        )


print("\n" + "=" * 75)
print("ANALYSIS COMPLETE")
print("=" * 75)