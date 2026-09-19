from pathlib import Path
from collections import Counter
import numpy as np

DATASET_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]


# --------------------------------------------------
# ANALYZE EACH SPLIT
# --------------------------------------------------

for split in SPLITS:

    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    image_files = []

    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]:
        image_files.extend(image_dir.glob(ext))

    label_files = list(label_dir.glob("*.txt"))

    objects_per_image = []
    class_counter = Counter()

    images_with_small_objects = 0
    images_with_medium_objects = 0
    images_with_large_objects = 0

    for label_file in label_files:

        object_count = 0

        has_small = False
        has_medium = False
        has_large = False

        with open(label_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                class_id, x, y, w, h = map(float, parts)

                object_count += 1

                class_counter[int(class_id)] += 1

                area = w * h

                if area < 0.01:
                    has_small = True

                elif area < 0.05:
                    has_medium = True

                else:
                    has_large = True

        objects_per_image.append(object_count)

        if has_small:
            images_with_small_objects += 1

        if has_medium:
            images_with_medium_objects += 1

        if has_large:
            images_with_large_objects += 1


    # --------------------------------------------------
    # RESULTS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print(f"{split.upper()} SPLIT")
    print("=" * 70)

    print(f"Images              : {len(image_files)}")
    print(f"Label files         : {len(label_files)}")
    print(f"Total objects       : {sum(objects_per_image)}")

    print("\nObjects per image")

    print(f"  Minimum           : {min(objects_per_image)}")
    print(f"  Median            : {np.median(objects_per_image):.2f}")
    print(f"  Mean              : {np.mean(objects_per_image):.2f}")
    print(f"  Maximum           : {max(objects_per_image)}")

    print("\nImages containing:")

    print(
        f"  Small objects     : "
        f"{images_with_small_objects} "
        f"({images_with_small_objects / len(image_files) * 100:.2f}%)"
    )

    print(
        f"  Medium objects    : "
        f"{images_with_medium_objects} "
        f"({images_with_medium_objects / len(image_files) * 100:.2f}%)"
    )

    print(
        f"  Large objects     : "
        f"{images_with_large_objects} "
        f"({images_with_large_objects / len(image_files) * 100:.2f}%)"
    )

    print("\nNumber of unique classes:")

    print(f"  {len(class_counter)}")

    print("\nTop 10 classes:")

    for class_id, count in class_counter.most_common(10):

        print(
            f"  Class {class_id:3d} : {count:4d}"
        )


print("\n" + "=" * 70)
print("SPLIT DISTRIBUTION ANALYSIS COMPLETE")
print("=" * 70)