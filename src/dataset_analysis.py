from pathlib import Path
import yaml
from collections import Counter
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]


# ============================================================
# LOAD DATA.YAML
# ============================================================

yaml_path = DATASET_DIR / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

class_names = data["names"]

print("\n" + "=" * 60)
print("ROAD SIGN DATASET ANALYSIS")
print("=" * 60)

print(f"\nNumber of classes: {len(class_names)}")

print("\nClasses:")
for class_id, class_name in enumerate(class_names):
    print(f"{class_id:3} -> {class_name}")


# ============================================================
# DATASET STATISTICS
# ============================================================

class_counts = Counter()
split_statistics = {}

total_images = 0
total_annotations = 0


for split in SPLITS:

    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    image_files = list(images_dir.glob("*"))
    label_files = list(labels_dir.glob("*.txt"))

    split_images = 0
    split_annotations = 0

    for image_path in image_files:

        # Only process actual image files
        if image_path.suffix.lower() not in [
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        ]:
            continue

        split_images += 1

        label_path = labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():

            with open(label_path, "r", encoding="utf-8") as file:

                for line in file:

                    line = line.strip()

                    if not line:
                        continue

                    parts = line.split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])

                    class_counts[class_id] += 1

                    split_annotations += 1

    split_statistics[split] = {
        "images": split_images,
        "labels": len(label_files),
        "annotations": split_annotations
    }

    total_images += split_images
    total_annotations += split_annotations


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

for split, stats in split_statistics.items():

    print(f"\n{split.upper()}")

    print(f"Images       : {stats['images']}")
    print(f"Label files  : {stats['labels']}")
    print(f"Annotations  : {stats['annotations']}")

print("\n" + "-" * 60)

print(f"Total images       : {total_images}")
print(f"Total annotations  : {total_annotations}")
print(f"Total classes      : {len(class_names)}")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

for class_id in range(len(class_names)):

    class_name = class_names[class_id]
    count = class_counts[class_id]

    print(f"{class_id:3} | {count:5} | {class_name}")


# ============================================================
# CREATE RESULTS DIRECTORY
# ============================================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)


# ============================================================
# CLASS DISTRIBUTION GRAPH
# ============================================================

class_ids = list(range(len(class_names)))
counts = [class_counts[i] for i in class_ids]

plt.figure(figsize=(18, 8))

plt.bar(class_ids, counts)

plt.xlabel("Class ID")
plt.ylabel("Number of Objects")
plt.title("Traffic Sign Class Distribution")

plt.tight_layout()

plt.savefig(
    results_dir / "class_distribution.png",
    dpi=200
)

plt.close()

print("\nSaved:")
print("results/class_distribution.png")


# ============================================================
# SPLIT DISTRIBUTION GRAPH
# ============================================================

split_names = list(split_statistics.keys())
image_counts = [
    split_statistics[s]["images"]
    for s in split_names
]

plt.figure(figsize=(8, 5))

plt.bar(split_names, image_counts)

plt.xlabel("Dataset Split")
plt.ylabel("Number of Images")
plt.title("Images per Dataset Split")

plt.tight_layout()

plt.savefig(
    results_dir / "dataset_split_distribution.png",
    dpi=200
)

plt.close()

print("results/dataset_split_distribution.png")


print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)