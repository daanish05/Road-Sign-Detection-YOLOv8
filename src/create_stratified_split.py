from pathlib import Path
from collections import defaultdict, Counter
import shutil
import random


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

TRAIN_IMAGES = DATASET_DIR / "train" / "images"
TRAIN_LABELS = DATASET_DIR / "train" / "labels"

VALID_IMAGES = DATASET_DIR / "valid" / "images"
VALID_LABELS = DATASET_DIR / "valid" / "labels"

OUTPUT_ROOT = DATASET_DIR / "stratified"

OUTPUT_TRAIN_IMAGES = OUTPUT_ROOT / "train" / "images"
OUTPUT_TRAIN_LABELS = OUTPUT_ROOT / "train" / "labels"

OUTPUT_VALID_IMAGES = OUTPUT_ROOT / "valid" / "images"
OUTPUT_VALID_LABELS = OUTPUT_ROOT / "valid" / "labels"

VALID_RATIO = 0.20
RANDOM_SEED = 42


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
# COLLECT IMAGE / LABEL PAIRS
# ============================================================

def collect_pairs(image_dir, label_dir):

    pairs = []

    for image_path in image_dir.iterdir():

        if not image_path.is_file():
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        if not label_path.exists():

            raise RuntimeError(
                f"Missing label for image:\n{image_path}"
            )

        pairs.append(
            (image_path, label_path)
        )

    return pairs


train_pairs = collect_pairs(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

valid_pairs = collect_pairs(
    VALID_IMAGES,
    VALID_LABELS
)

all_pairs = train_pairs + valid_pairs


# ============================================================
# READ CLASSES FROM LABEL FILE
# ============================================================

def read_classes(label_path):

    classes = set()

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

            classes.add(class_id)

    return classes


# ============================================================
# IMAGE -> CLASSES
# ============================================================

image_classes = {}

for image_path, label_path in all_pairs:

    image_classes[image_path.name] = read_classes(
        label_path
    )


# ============================================================
# GROUP IMAGES BY SOURCE
# ============================================================

groups = defaultdict(list)

for image_path, label_path in all_pairs:

    source = get_source_stem(
        image_path.name
    )

    groups[source].append(
        (image_path, label_path)
    )


# ============================================================
# GROUP -> CLASSES
# ============================================================

group_classes = {}

for source, items in groups.items():

    classes = set()

    for image_path, label_path in items:

        classes.update(
            image_classes[image_path.name]
        )

    group_classes[source] = classes


# ============================================================
# COUNT TOTAL IMAGE SUPPORT FOR EACH CLASS
# ============================================================

class_image_count = Counter()

for image_path, label_path in all_pairs:

    for class_id in image_classes[
        image_path.name
    ]:

        class_image_count[class_id] += 1


# ============================================================
# COUNT CLASS SUPPORT INSIDE EACH GROUP
# ============================================================

group_class_counts = {}

for source, items in groups.items():

    counts = Counter()

    for image_path, label_path in items:

        for class_id in image_classes[
            image_path.name
        ]:

            counts[class_id] += 1

    group_class_counts[source] = counts


# ============================================================
# TARGET VALIDATION SIZE
# ============================================================

total_images = len(all_pairs)

target_valid_images = round(
    total_images * VALID_RATIO
)


# ============================================================
# INFORMATION
# ============================================================

single_image_classes = sorted(
    class_id
    for class_id, count in class_image_count.items()
    if count == 1
)


print("=" * 80)
print("CREATING SOURCE-GROUP-AWARE STRATIFIED SPLIT")
print("=" * 80)

print(
    f"\nTotal images available: {total_images}"
)

print(
    f"Total source groups: {len(groups)}"
)

print(
    f"Target validation images: {target_valid_images}"
)

print(
    f"Validation ratio: {VALID_RATIO * 100:.2f}%"
)

print(
    f"Classes appearing in only one image: "
    f"{len(single_image_classes)}"
)

if single_image_classes:

    print(
        f"Single-image classes: "
        f"{single_image_classes}"
    )


# ============================================================
# CHECK WHETHER A GROUP CAN GO TO VALIDATION
# ============================================================

def can_move_to_validation(source, current_valid_groups):

    group_counts = group_class_counts[source]

    # Calculate how many images currently remain
    # in training for each class.

    remaining_training = Counter()

    for other_source, items in groups.items():

        if other_source == source:
            continue

        if other_source in current_valid_groups:
            continue

        for image_path, label_path in items:

            for class_id in image_classes[
                image_path.name
            ]:

                remaining_training[class_id] += 1


    # Every class present in this candidate group
    # must still have at least one training image.

    for class_id in group_counts:

        if remaining_training[class_id] <= 0:

            return False

    return True


# ============================================================
# INITIALIZE
# ============================================================

random.seed(RANDOM_SEED)

all_group_names = list(groups.keys())

random.shuffle(
    all_group_names
)

selected_valid_groups = set()

validation_classes = set()


# ============================================================
# HELPER: CURRENT VALIDATION SIZE
# ============================================================

def current_validation_size():

    return sum(
        len(groups[source])
        for source in selected_valid_groups
    )


# ============================================================
# STRATIFIED SELECTION
# ============================================================

remaining_groups = set(
    all_group_names
)


while True:

    current_size = current_validation_size()

    if current_size >= target_valid_images:
        break

    candidates = []

    for source in remaining_groups:

        group_size = len(
            groups[source]
        )

        # Don't exceed target.
        if (
            current_size + group_size
            > target_valid_images
        ):
            continue

        # Critical class-support check.
        if not can_move_to_validation(
            source,
            selected_valid_groups
        ):
            continue

        new_classes = (
            group_classes[source]
            - validation_classes
        )

        candidates.append(
            (
                len(new_classes),
                -group_size,
                source
            )
        )


    # No more legal groups.
    if not candidates:
        break


    # Prefer:
    # 1. Groups introducing new classes
    # 2. Smaller groups when tied

    candidates.sort(
        key=lambda item: (
            -item[0],
            -item[1]
        )
    )


    selected_source = candidates[0][2]


    selected_valid_groups.add(
        selected_source
    )

    validation_classes.update(
        group_classes[selected_source]
    )

    remaining_groups.remove(
        selected_source
    )


# ============================================================
# BUILD TRAIN / VALID LISTS
# ============================================================

train_items = []
valid_items = []


for source, items in groups.items():

    if source in selected_valid_groups:

        valid_items.extend(
            items
        )

    else:

        train_items.extend(
            items
        )


# ============================================================
# SHUFFLE
# ============================================================

random.shuffle(
    train_items
)

random.shuffle(
    valid_items
)


# ============================================================
# FINAL IMAGE OVERLAP CHECK
# ============================================================

train_image_names = {
    image_path.name
    for image_path, label_path in train_items
}

valid_image_names = {
    image_path.name
    for image_path, label_path in valid_items
}


image_overlap = (
    train_image_names
    &
    valid_image_names
)


if image_overlap:

    raise RuntimeError(
        "ERROR: Image overlap detected!"
    )


# ============================================================
# FINAL SOURCE-GROUP CHECK
# ============================================================

train_source_groups = {
    get_source_stem(name)
    for name in train_image_names
}

valid_source_groups = {
    get_source_stem(name)
    for name in valid_image_names
}


source_group_overlap = (
    train_source_groups
    &
    valid_source_groups
)


print("\n" + "=" * 80)
print("FINAL SAFETY CHECK")
print("=" * 80)

print(
    f"\nTrain source groups: "
    f"{len(train_source_groups)}"
)

print(
    f"Valid source groups: "
    f"{len(valid_source_groups)}"
)

print(
    f"Overlapping source groups: "
    f"{len(source_group_overlap)}"
)


if source_group_overlap:

    print(
        "\nFirst overlapping groups:"
    )

    for source in sorted(
        source_group_overlap
    )[:20]:

        print(
            f"  {source}"
        )

    raise RuntimeError(
        "ERROR: Source-group leakage detected!"
    )


# ============================================================
# FINAL CLASS COVERAGE CHECK
# ============================================================

train_class_counts = Counter()
valid_class_counts = Counter()


for image_path, label_path in train_items:

    for class_id in image_classes[
        image_path.name
    ]:

        train_class_counts[class_id] += 1


for image_path, label_path in valid_items:

    for class_id in image_classes[
        image_path.name
    ]:

        valid_class_counts[class_id] += 1


validation_without_training = sorted(
    set(valid_class_counts)
    -
    set(train_class_counts)
)


print(
    f"\nTrain classes: "
    f"{len(train_class_counts)}"
)

print(
    f"Valid classes: "
    f"{len(valid_class_counts)}"
)

print(
    f"Validation classes without training: "
    f"{len(validation_without_training)}"
)


if validation_without_training:

    print(
        "\nPROBLEMATIC CLASSES:"
    )

    for class_id in validation_without_training:

        print(
            f"Class {class_id}: "
            f"valid={valid_class_counts[class_id]}, "
            f"train={train_class_counts[class_id]}"
        )

    raise RuntimeError(
        "ERROR: Validation contains classes "
        "with zero training examples!"
    )


# ============================================================
# PREPARE OUTPUT
# ============================================================

if OUTPUT_ROOT.exists():

    print(
        "\nRemoving previous stratified split..."
    )

    shutil.rmtree(
        OUTPUT_ROOT
    )


OUTPUT_TRAIN_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_TRAIN_LABELS.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VALID_IMAGES.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_VALID_LABELS.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# COPY FILES
# ============================================================

def copy_items(
    items,
    image_destination,
    label_destination
):

    for image_path, label_path in items:

        shutil.copy2(
            image_path,
            image_destination / image_path.name
        )

        shutil.copy2(
            label_path,
            label_destination / label_path.name
        )


copy_items(
    train_items,
    OUTPUT_TRAIN_IMAGES,
    OUTPUT_TRAIN_LABELS
)

copy_items(
    valid_items,
    OUTPUT_VALID_IMAGES,
    OUTPUT_VALID_LABELS
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("STRATIFIED SPLIT CREATED SUCCESSFULLY")
print("=" * 80)

print(
    f"\nTotal images: "
    f"{total_images}"
)

print(
    f"Train images: "
    f"{len(train_items)}"
)

print(
    f"Valid images: "
    f"{len(valid_items)}"
)

print(
    f"Validation ratio: "
    f"{len(valid_items) / total_images * 100:.2f}%"
)

print(
    f"\nTrain source groups: "
    f"{len(train_source_groups)}"
)

print(
    f"Valid source groups: "
    f"{len(valid_source_groups)}"
)

print(
    f"Overlapping source groups: "
    f"{len(source_group_overlap)}"
)

print(
    f"\nTrain classes: "
    f"{len(train_class_counts)}"
)

print(
    f"Valid classes: "
    f"{len(valid_class_counts)}"
)

print(
    f"Validation classes without training: "
    f"{len(validation_without_training)}"
)


print("\n" + "=" * 80)
print("SAFETY CHECKS")
print("=" * 80)

print("PASS: No image overlap.")
print("PASS: No source-group leakage.")
print(
    "PASS: Every validation class has "
    "training support."
)


print("\nOutput directories:")

print(
    f"  {OUTPUT_TRAIN_IMAGES}"
)

print(
    f"  {OUTPUT_TRAIN_LABELS}"
)

print(
    f"  {OUTPUT_VALID_IMAGES}"
)

print(
    f"  {OUTPUT_VALID_LABELS}"
)


print(
    "\nOriginal train/valid/test directories "
    "were NOT modified."
)

print(
    "Original test set was NOT modified."
)

print("\nDONE")