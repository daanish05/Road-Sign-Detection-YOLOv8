from pathlib import Path
import yaml
from collections import Counter

DATASET_DIR = Path("dataset")
YAML_FILE = DATASET_DIR / "data.yaml"

print("=" * 70)
print("FINAL DATASET INTEGRITY CHECK")
print("=" * 70)

# ---------------------------------------------------------
# 1. Load YAML
# ---------------------------------------------------------

with open(YAML_FILE, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

names = data.get("names", [])
nc = data.get("nc")

print("\n[1] YAML CONFIGURATION")
print("-" * 70)
print(f"nc: {nc}")
print(f"Number of names: {len(names)}")

if nc == len(names):
    print("PASS: nc matches number of class names")
else:
    print("FAIL: nc does not match number of class names")


# ---------------------------------------------------------
# 2. Check dataset directories
# ---------------------------------------------------------

print("\n[2] DATASET DIRECTORIES")
print("-" * 70)

for split in ["train", "valid", "test"]:
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    print(f"\n{split}:")
    print(f"  Images directory: {image_dir} -> {'OK' if image_dir.exists() else 'MISSING'}")
    print(f"  Labels directory: {label_dir} -> {'OK' if label_dir.exists() else 'MISSING'}")

    if image_dir.exists():
        images = list(image_dir.glob("*"))
        print(f"  Images found: {len(images)}")

    if label_dir.exists():
        labels = list(label_dir.glob("*.txt"))
        print(f"  Label files found: {len(labels)}")


# ---------------------------------------------------------
# 3. Validate class IDs
# ---------------------------------------------------------

print("\n[3] CLASS ID VALIDATION")
print("-" * 70)

class_counts = Counter()
invalid_class_ids = []

for split in ["train", "valid", "test"]:
    label_dir = DATASET_DIR / split / "labels"

    if not label_dir.exists():
        continue

    for label_file in label_dir.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line_number, line in enumerate(lines, start=1):

            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                invalid_class_ids.append(
                    (str(label_file), line_number, parts[0])
                )
                continue

            class_counts[class_id] += 1

            if class_id < 0 or class_id >= len(names):
                invalid_class_ids.append(
                    (str(label_file), line_number, class_id)
                )


print(f"Unique class IDs found: {len(class_counts)}")
print(f"Minimum class ID: {min(class_counts) if class_counts else 'N/A'}")
print(f"Maximum class ID: {max(class_counts) if class_counts else 'N/A'}")

if invalid_class_ids:
    print(f"FAIL: {len(invalid_class_ids)} invalid class IDs found")

    for item in invalid_class_ids[:10]:
        print(" ", item)
else:
    print("PASS: All class IDs are within 0 to", len(names) - 1)


# ---------------------------------------------------------
# 4. Check missing image/label pairs
# ---------------------------------------------------------

print("\n[4] IMAGE / LABEL PAIR CHECK")
print("-" * 70)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ["train", "valid", "test"]:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        continue

    images = {
        p.stem
        for p in image_dir.iterdir()
        if p.suffix.lower() in image_extensions
    }

    labels = {
        p.stem
        for p in label_dir.glob("*.txt")
    }

    images_without_labels = images - labels
    labels_without_images = labels - images

    print(f"\n{split}:")
    print(f"  Images without labels: {len(images_without_labels)}")
    print(f"  Labels without images: {len(labels_without_images)}")

    if images_without_labels:
        print("  Example:", list(images_without_labels)[:5])

    if labels_without_images:
        print("  Example:", list(labels_without_images)[:5])


# ---------------------------------------------------------
# 5. Class distribution
# ---------------------------------------------------------

print("\n[5] CLASS DISTRIBUTION")
print("-" * 70)

print(f"Total annotations: {sum(class_counts.values())}")
print(f"Total classes used: {len(class_counts)}")

print("\nTop 15 classes by annotation count:")

for class_id, count in class_counts.most_common(15):
    print(
        f"  ID {class_id:3d} | "
        f"Count: {count:4d} | "
        f"Name: {names[class_id]}"
    )

print("\nClasses with <= 3 annotations:")

rare_classes = [
    (class_id, count)
    for class_id, count in class_counts.items()
    if count <= 3
]

for class_id, count in sorted(rare_classes):
    print(
        f"  ID {class_id:3d} | "
        f"Count: {count:2d} | "
        f"Name: {names[class_id]}"
    )

print(f"\nNumber of classes with <= 3 annotations: {len(rare_classes)}")


# ---------------------------------------------------------
# 6. Final result
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

if (
    nc == len(names)
    and not invalid_class_ids
):
    print("DATASET STRUCTURE: PASS")
else:
    print("DATASET STRUCTURE: NEEDS ATTENTION")

print("=" * 70)