from pathlib import Path
from collections import Counter

DATASET = Path("dataset")

SPLITS = {
    "train": DATASET / "train" / "labels",
    "valid": DATASET / "valid" / "labels",
    "test": DATASET / "test" / "labels",
}

NUM_CLASSES = 264


def count_classes(label_dir):

    counts = Counter()

    for label_file in label_dir.glob("*.txt"):

        with open(label_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) == 5:

                    class_id = int(parts[0])

                    counts[class_id] += 1

    return counts


print("=" * 80)
print("CLASS COVERAGE ANALYSIS")
print("=" * 80)

all_counts = {}

for split, label_dir in SPLITS.items():

    counts = count_classes(label_dir)

    all_counts[split] = counts

    print(f"\n{split.upper()}")
    print("-" * 40)

    print(f"Classes represented : {len(counts)}")
    print(f"Total annotations   : {sum(counts.values())}")


# ============================================================
# CLASS COVERAGE TABLE
# ============================================================

print("\n" + "=" * 80)
print("CLASSES WITH TRAIN / VALID / TEST COVERAGE")
print("=" * 80)

print(
    f"\n{'ID':>5} "
    f"{'TRAIN':>8} "
    f"{'VALID':>8} "
    f"{'TEST':>8}"
)

print("-" * 40)

for class_id in range(NUM_CLASSES):

    train = all_counts["train"].get(class_id, 0)
    valid = all_counts["valid"].get(class_id, 0)
    test = all_counts["test"].get(class_id, 0)

    print(
        f"{class_id:>5} "
        f"{train:>8} "
        f"{valid:>8} "
        f"{test:>8}"
    )


# ============================================================
# IMPORTANT CATEGORIES
# ============================================================

train_only_missing = []
valid_without_train = []
test_without_train = []

for class_id in range(NUM_CLASSES):

    train = all_counts["train"].get(class_id, 0)
    valid = all_counts["valid"].get(class_id, 0)
    test = all_counts["test"].get(class_id, 0)

    if valid > 0 and train == 0:
        valid_without_train.append(class_id)

    if test > 0 and train == 0:
        test_without_train.append(class_id)

    if train > 0 and valid == 0 and test == 0:
        train_only_missing.append(class_id)


print("\n" + "=" * 80)
print("CRITICAL COVERAGE PROBLEMS")
print("=" * 80)

print("\nValidation classes with ZERO training examples:")

if valid_without_train:
    print(valid_without_train)
else:
    print("None")


print("\nTest classes with ZERO training examples:")

if test_without_train:
    print(test_without_train)
else:
    print("None")


print("\nTraining classes absent from BOTH validation and test:")

if train_only_missing:
    print(train_only_missing)
else:
    print("None")


# ============================================================
# TRAINING RARITY
# ============================================================

print("\n" + "=" * 80)
print("TRAINING CLASS FREQUENCY")
print("=" * 80)

frequency_groups = {
    "0": [],
    "1-5": [],
    "6-10": [],
    "11-20": [],
    "21-50": [],
    "51-100": [],
    "101+": [],
}

for class_id in range(NUM_CLASSES):

    count = all_counts["train"].get(class_id, 0)

    if count == 0:
        frequency_groups["0"].append(class_id)
    elif count <= 5:
        frequency_groups["1-5"].append(class_id)
    elif count <= 10:
        frequency_groups["6-10"].append(class_id)
    elif count <= 20:
        frequency_groups["11-20"].append(class_id)
    elif count <= 50:
        frequency_groups["21-50"].append(class_id)
    elif count <= 100:
        frequency_groups["51-100"].append(class_id)
    else:
        frequency_groups["101+"].append(class_id)


for group, classes in frequency_groups.items():

    print(
        f"{group:>8} training instances : "
        f"{len(classes)} classes"
    )

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)