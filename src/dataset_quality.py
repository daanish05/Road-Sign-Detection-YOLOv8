from pathlib import Path
from PIL import Image
import numpy as np
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

SPLITS = [
    "train",
    "valid",
    "test"
]


# ============================================================
# BLACK PIXEL DETECTION
# ============================================================

# A pixel is considered "black" if all RGB channels
# are below this threshold.
BLACK_THRESHOLD = 10

# Percentage of black pixels above which we flag an image.
BLACK_PIXEL_RATIO_THRESHOLD = 0.01


# ============================================================
# OUTPUT
# ============================================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

report_path = results_dir / "dataset_quality_report.txt"


# ============================================================
# COUNTERS
# ============================================================

total_images = 0
problematic_images = 0

total_labels = 0
invalid_labels = 0
missing_labels = 0

black_image_results = []


# ============================================================
# ANALYZE DATASET
# ============================================================

with open(report_path, "w", encoding="utf-8") as report:

    report.write("ROAD SIGN DATASET QUALITY AUDIT\n")
    report.write("=" * 70 + "\n\n")

    for split in SPLITS:

        images_dir = DATASET_DIR / split / "images"
        labels_dir = DATASET_DIR / split / "labels"

        report.write(f"\n{split.upper()}\n")
        report.write("-" * 70 + "\n")

        if not images_dir.exists():

            report.write(
                f"Images directory not found: {images_dir}\n"
            )

            continue

        image_files = [
            f for f in images_dir.iterdir()
            if f.suffix.lower() in [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            ]
        ]

        split_problematic = 0

        for image_path in image_files:

            total_images += 1

            # ------------------------------------------------
            # Open image
            # ------------------------------------------------

            try:

                image = Image.open(image_path).convert("RGB")

                image_array = np.array(image)

            except Exception as error:

                report.write(
                    f"ERROR reading image: "
                    f"{image_path.name} | {error}\n"
                )

                continue

            # ------------------------------------------------
            # Detect black pixels
            # ------------------------------------------------

            black_pixels = np.all(
                image_array <= BLACK_THRESHOLD,
                axis=2
            )

            black_ratio = black_pixels.mean()

            # ------------------------------------------------
            # Label file
            # ------------------------------------------------

            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.exists():

                missing_labels += 1

                report.write(
                    f"MISSING LABEL: {image_path.name}\n"
                )

            else:

                total_labels += 1

                # --------------------------------------------
                # Validate YOLO annotations
                # --------------------------------------------

                try:

                    with open(
                        label_path,
                        "r",
                        encoding="utf-8"
                    ) as label_file:

                        lines = label_file.readlines()

                    for line_number, line in enumerate(
                        lines,
                        start=1
                    ):

                        line = line.strip()

                        if not line:
                            continue

                        parts = line.split()

                        if len(parts) != 5:

                            invalid_labels += 1

                            report.write(
                                f"INVALID LABEL: "
                                f"{label_path.name} "
                                f"line {line_number}: "
                                f"{line}\n"
                            )

                            continue

                        try:

                            class_id = int(parts[0])

                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])

                        except ValueError:

                            invalid_labels += 1

                            report.write(
                                f"INVALID NUMERIC LABEL: "
                                f"{label_path.name} "
                                f"line {line_number}\n"
                            )

                            continue

                        # YOLO normalized coordinates
                        values = [
                            x_center,
                            y_center,
                            width,
                            height
                        ]

                        # Values should be between 0 and 1
                        if not all(
                            0 <= value <= 1
                            for value in values
                        ):

                            invalid_labels += 1

                            report.write(
                                f"OUT OF RANGE: "
                                f"{label_path.name} "
                                f"line {line_number}: "
                                f"{line}\n"
                            )

                        if width <= 0 or height <= 0:

                            invalid_labels += 1

                            report.write(
                                f"INVALID BOX SIZE: "
                                f"{label_path.name} "
                                f"line {line_number}: "
                                f"{line}\n"
                            )

                except Exception as error:

                    invalid_labels += 1

                    report.write(
                        f"ERROR READING LABEL: "
                        f"{label_path.name} | {error}\n"
                    )

            # ------------------------------------------------
            # Record black artifact
            # ------------------------------------------------

            if black_ratio >= BLACK_PIXEL_RATIO_THRESHOLD:

                problematic_images += 1
                split_problematic += 1

                black_image_results.append(
                    (
                        split,
                        image_path.name,
                        black_ratio
                    )
                )

        report.write(
            f"\nImages: {len(image_files)}\n"
        )

        report.write(
            f"Images with significant black pixels: "
            f"{split_problematic}\n"
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    report.write("\n\n")
    report.write("=" * 70 + "\n")
    report.write("SUMMARY\n")
    report.write("=" * 70 + "\n")

    report.write(
        f"Total images: {total_images}\n"
    )

    report.write(
        f"Images with black artifacts: "
        f"{problematic_images}\n"
    )

    report.write(
        f"Label files found: {total_labels}\n"
    )

    report.write(
        f"Missing labels: {missing_labels}\n"
    )

    report.write(
        f"Invalid annotation problems: "
        f"{invalid_labels}\n"
    )

    # ========================================================
    # BLACK ARTIFACT LIST
    # ========================================================

    report.write("\n")
    report.write("=" * 70 + "\n")
    report.write("IMAGES WITH BLACK PIXEL ARTIFACTS\n")
    report.write("=" * 70 + "\n")

    for split, filename, ratio in sorted(
        black_image_results,
        key=lambda x: x[2],
        reverse=True
    ):

        report.write(
            f"{split:6} | "
            f"{ratio * 100:7.3f}% | "
            f"{filename}\n"
        )


print("\n" + "=" * 60)
print("DATASET QUALITY AUDIT COMPLETE")
print("=" * 60)

print(f"\nTotal images: {total_images}")

print(
    f"Images with black artifacts: "
    f"{problematic_images}"
)

print(
    f"Missing labels: "
    f"{missing_labels}"
)

print(
    f"Invalid annotation problems: "
    f"{invalid_labels}"
)

print(f"\nFull report:")
print(report_path)