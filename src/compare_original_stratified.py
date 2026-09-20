from pathlib import Path
from collections import Counter


# ============================================================
# DIRECTORIES
# ============================================================

ORIGINAL_DIR = Path("dataset")
STRATIFIED_DIR = Path("dataset/stratified")


# ============================================================
# COUNT ANNOTATIONS
# ============================================================

def count_annotations(label_dir):

    total = 0
    class_counts = Counter()
    files = {}

    for label_path in label_dir.glob("*.txt"):

        count = 0

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                parts = line.strip().split()

                if not parts:
                    continue

                class_id = int(parts[0])

                class_counts[class_id] += 1
                count += 1

        files[label_path.name] = count
        total += count

    return total, class_counts, files


# ============================================================
# ORIGINAL
# ============================================================

original_train_total, original_train_classes, original_train_files = (
    count_annotations(
        ORIGINAL_DIR / "train" / "labels"
    )
)

original_valid_total, original_valid_classes, original_valid_files = (
    count_annotations(
        ORIGINAL_DIR / "valid" / "labels"
    )
)


# ============================================================
# STRATIFIED
# ============================================================

stratified_train_total, stratified_train_classes, stratified_train_files = (
    count_annotations(
        STRATIFIED_DIR / "train" / "labels"
    )
)

stratified_valid_total, stratified_valid_classes, stratified_valid_files = (
    count_annotations(
        STRATIFIED_DIR / "valid" / "labels"
    )
)


# ============================================================
# TOTALS
# ============================================================

original_total = (
    original_train_total +
    original_valid_total
)

stratified_total = (
    stratified_train_total +
    stratified_valid_total
)


print("=" * 80)
print("ORIGINAL vs STRATIFIED ANNOTATION CHECK")
print("=" * 80)

print(
    f"\nOriginal train annotations     : "
    f"{original_train_total}"
)

print(
    f"Original valid annotations     : "
    f"{original_valid_total}"
)

print(
    f"Original train+valid total     : "
    f"{original_total}"
)

print(
    f"\nStratified train annotations   : "
    f"{stratified_train_total}"
)

print(
    f"Stratified valid annotations   : "
    f"{stratified_valid_total}"
)

print(
    f"Stratified train+valid total   : "
    f"{stratified_total}"
)

print(
    f"\nDifference                     : "
    f"{original_total - stratified_total}"
)


# ============================================================
# CLASS DISTRIBUTION COMPARISON
# ============================================================

original_classes = (
    original_train_classes +
    original_valid_classes
)

stratified_classes = (
    stratified_train_classes +
    stratified_valid_classes
)


print("\n" + "=" * 80)
print("CLASS DISTRIBUTION DIFFERENCES")
print("=" * 80)

differences = []

all_classes = (
    set(original_classes) |
    set(stratified_classes)
)


for class_id in sorted(all_classes):

    original_count = original_classes[class_id]
    stratified_count = stratified_classes[class_id]

    if original_count != stratified_count:

        differences.append(
            (
                class_id,
                original_count,
                stratified_count,
                original_count - stratified_count
            )
        )


if not differences:

    print(
        "\nPASS: Class distributions are identical."
    )

else:

    print(
        f"\nClasses with differences: "
        f"{len(differences)}"
    )

    print(
        "\nClass | Original | Stratified | Difference"
    )

    print("-" * 55)

    for (
        class_id,
        original_count,
        stratified_count,
        difference
    ) in differences:

        print(
            f"{class_id:5d} | "
            f"{original_count:8d} | "
            f"{stratified_count:10d} | "
            f"{difference:10d}"
        )


# ============================================================
# FILE-LEVEL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("FILE-LEVEL ANNOTATION CHECK")
print("=" * 80)


original_files = {}

original_files.update(
    original_train_files
)

original_files.update(
    original_valid_files
)


stratified_files = {}

stratified_files.update(
    stratified_train_files
)

stratified_files.update(
    stratified_valid_files
)


missing_from_stratified = []

different_counts = []


for filename, original_count in original_files.items():

    if filename not in stratified_files:

        missing_from_stratified.append(
            filename
        )

    else:

        stratified_count = (
            stratified_files[filename]
        )

        if original_count != stratified_count:

            different_counts.append(
                (
                    filename,
                    original_count,
                    stratified_count,
                    original_count - stratified_count
                )
            )


print(
    f"\nOriginal label files: "
    f"{len(original_files)}"
)

print(
    f"Stratified label files: "
    f"{len(stratified_files)}"
)

print(
    f"Missing label files in stratified: "
    f"{len(missing_from_stratified)}"
)

print(
    f"Files with different annotation counts: "
    f"{len(different_counts)}"
)


if missing_from_stratified:

    print(
        "\nMISSING FILES:"
    )

    for filename in missing_from_stratified:

        print(
            f"  {filename}"
        )


if different_counts:

    print(
        "\nFILES WITH DIFFERENT ANNOTATION COUNTS:"
    )

    for (
        filename,
        original_count,
        stratified_count,
        difference
    ) in different_counts:

        print(
            f"  {filename}"
            f" | original={original_count}"
            f" | stratified={stratified_count}"
            f" | difference={difference}"
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 80)
print("RESULT")
print("=" * 80)

if (
    original_total == stratified_total
    and not missing_from_stratified
    and not different_counts
):

    print(
        "\nPASS: Stratified split preserves "
        "all original annotations."
    )

else:

    print(
        "\nWARNING: Annotation discrepancy detected."
    )

    print(
        "DO NOT TRAIN YET."
    )