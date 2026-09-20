from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("dataset/stratified")

TRAIN_IMAGES = DATASET_ROOT / "train" / "images"
TRAIN_LABELS = DATASET_ROOT / "train" / "labels"

VALID_IMAGES = DATASET_ROOT / "valid" / "images"
VALID_LABELS = DATASET_ROOT / "valid" / "labels"


# ============================================================
# SOURCE GROUP
# SAME LOGIC AS analyze_source_groups.py
# ============================================================

def get_source_stem(filename):

    name = Path(filename).stem

    if ".rf." in name:
        return name.split(".rf.")[0]

    return name


# ============================================================
# COLLECT FILES
# ============================================================

def collect_images(image_dir):

    return {
        path.name
        for path in image_dir.iterdir()
        if path.is_file()
    }


def collect_labels(label_dir):

    return {
        path.name
        for path in label_dir.iterdir()
        if path.is_file()
    }


train_images = collect_images(
    TRAIN_IMAGES
)

valid_images = collect_images(
    VALID_IMAGES
)

train_labels = collect_labels(
    TRAIN_LABELS
)

valid_labels = collect_labels(
    VALID_LABELS
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STRATIFIED SPLIT INDEPENDENT VERIFICATION")
print("=" * 80)


# ============================================================
# IMAGE COUNTS
# ============================================================

print("\n" + "=" * 80)
print("IMAGE COUNTS")
print("=" * 80)

print(
    f"\nTrain images : {len(train_images)}"
)

print(
    f"Valid images : {len(valid_images)}"
)

print(
    f"Total images : "
    f"{len(train_images) + len(valid_images)}"
)

print(
    f"Validation ratio : "
    f"{len(valid_images) / (len(train_images) + len(valid_images)) * 100:.2f}%"
)


# ============================================================
# IMAGE OVERLAP
# ============================================================

print("\n" + "=" * 80)
print("IMAGE OVERLAP")
print("=" * 80)

image_overlap = (
    train_images &
    valid_images
)

print(
    f"\nOverlapping images: "
    f"{len(image_overlap)}"
)

if image_overlap:

    print("\nFirst overlaps:")

    for name in sorted(
        image_overlap
    )[:20]:

        print(name)

    raise RuntimeError(
        "ERROR: Image overlap detected!"
    )

else:

    print(
        "PASS: No image overlap."
    )


# ============================================================
# SOURCE GROUP ANALYSIS
# ============================================================

print("\n" + "=" * 80)
print("SOURCE GROUP ANALYSIS")
print("=" * 80)

train_groups = {
    get_source_stem(name)
    for name in train_images
}

valid_groups = {
    get_source_stem(name)
    for name in valid_images
}

group_overlap = (
    train_groups &
    valid_groups
)

print(
    f"\nTrain source groups : "
    f"{len(train_groups)}"
)

print(
    f"Valid source groups : "
    f"{len(valid_groups)}"
)

print(
    f"Overlapping source groups : "
    f"{len(group_overlap)}"
)

if group_overlap:

    print(
        "\nFirst overlapping groups:"
    )

    for group in sorted(
        group_overlap
    )[:20]:

        print(group)

    raise RuntimeError(
        "ERROR: Source-group leakage detected!"
    )

else:

    print(
        "PASS: No source-group leakage."
    )


# ============================================================
# IMAGE / LABEL PAIR CHECK
# ============================================================

print("\n" + "=" * 80)
print("IMAGE / LABEL PAIR CHECK")
print("=" * 80)


def expected_label_name(image_name):

    return (
        Path(image_name).stem
        + ".txt"
    )


missing_train_labels = [
    name
    for name in train_images
    if expected_label_name(name)
    not in train_labels
]

missing_valid_labels = [
    name
    for name in valid_images
    if expected_label_name(name)
    not in valid_labels
]


if missing_train_labels:

    print(
        "\nMissing train labels:"
    )

    for name in missing_train_labels[:20]:

        print(name)

    raise RuntimeError(
        "ERROR: Missing train labels!"
    )


if missing_valid_labels:

    print(
        "\nMissing validation labels:"
    )

    for name in missing_valid_labels[:20]:

        print(name)

    raise RuntimeError(
        "ERROR: Missing validation labels!"
    )


print(
    "PASS: Every image has a corresponding label."
)


# ============================================================
# READ CLASS IDS
# ============================================================

def read_class_counts(label_dir):

    counts = Counter()

    for label_path in label_dir.glob("*.txt"):

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                counts[class_id] += 1

    return counts


train_class_counts = read_class_counts(
    TRAIN_LABELS
)

valid_class_counts = read_class_counts(
    VALID_LABELS
)


# ============================================================
# CLASS COVERAGE
# ============================================================

print("\n" + "=" * 80)
print("CLASS COVERAGE")
print("=" * 80)

train_classes = set(
    train_class_counts
)

valid_classes = set(
    valid_class_counts
)

combined_classes = (
    train_classes |
    valid_classes
)

print(
    f"\nClasses in train : "
    f"{len(train_classes)}"
)

print(
    f"Classes in valid : "
    f"{len(valid_classes)}"
)

print(
    f"Combined classes : "
    f"{len(combined_classes)}"
)


validation_without_training = sorted(
    valid_classes -
    train_classes
)


print("\n" + "=" * 80)
print("VALIDATION CLASSES WITHOUT TRAINING SUPPORT")
print("=" * 80)

print(
    f"\nFound "
    f"{len(validation_without_training)} "
    f"validation classes with zero training examples."
)


if validation_without_training:

    for class_id in validation_without_training:

        print(
            f"Class {class_id}: "
            f"train=0, "
            f"valid={valid_class_counts[class_id]}"
        )

    raise RuntimeError(
        "ERROR: Validation contains classes "
        "without training support!"
    )

else:

    print(
        "PASS: Every validation class has "
        "training support."
    )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("CLASS DISTRIBUTION")
print("=" * 80)

print(
    f"\nTrain annotations : "
    f"{sum(train_class_counts.values())}"
)

print(
    f"Valid annotations : "
    f"{sum(valid_class_counts.values())}"
)

print(
    f"Total annotations : "
    f"{sum(train_class_counts.values()) + sum(valid_class_counts.values())}"
)


# ============================================================
# RARE CLASS INFORMATION
# ============================================================

combined_class_counts = (
    train_class_counts +
    valid_class_counts
)

rare_classes = {
    class_id: count
    for class_id, count
    in combined_class_counts.items()
    if count <= 5
}


print(
    f"\nClasses with <=5 total annotations: "
    f"{len(rare_classes)}"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("FINAL VERIFICATION RESULT")
print("=" * 80)

print(
    "\nPASS: Stratified split is independently verified."
)

print(
    "PASS: No image overlap."
)

print(
    "PASS: No source-group leakage."
)

print(
    "PASS: No missing image/label pairs."
)

print(
    "PASS: Every validation class has "
    "training support."
)

print(
    "\nOriginal dataset was not modified."
)

print(
    "Original test set was not checked or modified."
)

print("\nCHECK COMPLETE")