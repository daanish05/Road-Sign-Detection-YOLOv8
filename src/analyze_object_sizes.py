from pathlib import Path
import numpy as np

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]

IMAGE_SIZE = 640

# COCO-style area thresholds, expressed as a fraction
# of the full 640x640 image.
SMALL_THRESHOLD = 0.01
MEDIUM_THRESHOLD = 0.05


# --------------------------------------------------
# STORAGE
# --------------------------------------------------

all_widths = []
all_heights = []
all_areas = []

split_stats = {}

# --------------------------------------------------
# READ YOLO LABELS
# --------------------------------------------------

for split in SPLITS:

    labels_dir = DATASET_DIR / split / "labels"

    widths = []
    heights = []
    areas = []

    if not labels_dir.exists():
        print(f"WARNING: {labels_dir} does not exist")
        continue

    label_files = list(labels_dir.glob("*.txt"))

    for label_file in label_files:

        with open(label_file, "r") as f:

            for line in f:

                parts = line.strip().split()

                if len(parts) != 5:
                    continue

                try:
                    class_id, x_center, y_center, width, height = map(
                        float, parts
                    )
                except ValueError:
                    continue

                # YOLO width/height are normalized 0-1.
                # Convert to pixels.
                box_width = width * IMAGE_SIZE
                box_height = height * IMAGE_SIZE

                box_area = width * height

                widths.append(box_width)
                heights.append(box_height)
                areas.append(box_area)

                all_widths.append(box_width)
                all_heights.append(box_height)
                all_areas.append(box_area)

    split_stats[split] = {
        "objects": len(widths),
        "widths": widths,
        "heights": heights,
        "areas": areas,
    }


# --------------------------------------------------
# ANALYSIS FUNCTION
# --------------------------------------------------

def analyze_split(name, stats):

    widths = np.array(stats["widths"])
    heights = np.array(stats["heights"])
    areas = np.array(stats["areas"])

    if len(areas) == 0:
        return

    small = areas < SMALL_THRESHOLD
    medium = (areas >= SMALL_THRESHOLD) & (areas < MEDIUM_THRESHOLD)
    large = areas >= MEDIUM_THRESHOLD

    print("\n" + "=" * 70)
    print(f"{name.upper()} OBJECT SIZE ANALYSIS")
    print("=" * 70)

    print(f"Objects: {len(areas)}")

    print("\nBounding-box width (pixels)")
    print(f"  Minimum : {widths.min():.2f}")
    print(f"  Median  : {np.median(widths):.2f}")
    print(f"  Mean    : {widths.mean():.2f}")
    print(f"  Maximum : {widths.max():.2f}")

    print("\nBounding-box height (pixels)")
    print(f"  Minimum : {heights.min():.2f}")
    print(f"  Median  : {np.median(heights):.2f}")
    print(f"  Mean    : {heights.mean():.2f}")
    print(f"  Maximum : {heights.max():.2f}")

    print("\nBounding-box area")
    print(f"  Minimum : {areas.min() * 100:.4f}% of image")
    print(f"  Median  : {np.median(areas) * 100:.4f}% of image")
    print(f"  Mean    : {areas.mean() * 100:.4f}% of image")
    print(f"  Maximum : {areas.max() * 100:.4f}% of image")

    print("\nObject-size distribution")

    print(
        f"  Small   (<1% image)  : "
        f"{small.sum():4d} "
        f"({small.mean() * 100:.2f}%)"
    )

    print(
        f"  Medium  (1-5%)       : "
        f"{medium.sum():4d} "
        f"({medium.mean() * 100:.2f}%)"
    )

    print(
        f"  Large   (>=5%)       : "
        f"{large.sum():4d} "
        f"({large.mean() * 100:.2f}%)"
    )


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n" + "#" * 70)
print("ROAD SIGN DATASET - OBJECT SIZE ANALYSIS")
print("#" * 70)

for split in SPLITS:
    if split in split_stats:
        analyze_split(split, split_stats[split])


# --------------------------------------------------
# OVERALL DATASET
# --------------------------------------------------

all_widths = np.array(all_widths)
all_heights = np.array(all_heights)
all_areas = np.array(all_areas)

small = all_areas < SMALL_THRESHOLD
medium = (all_areas >= SMALL_THRESHOLD) & (all_areas < MEDIUM_THRESHOLD)
large = all_areas >= MEDIUM_THRESHOLD

print("\n" + "#" * 70)
print("OVERALL DATASET")
print("#" * 70)

print(f"\nTotal objects: {len(all_areas)}")

print("\nBounding-box width (pixels)")
print(f"  Minimum : {all_widths.min():.2f}")
print(f"  Median  : {np.median(all_widths):.2f}")
print(f"  Mean    : {all_widths.mean():.2f}")
print(f"  Maximum : {all_widths.max():.2f}")

print("\nBounding-box height (pixels)")
print(f"  Minimum : {all_heights.min():.2f}")
print(f"  Median  : {np.median(all_heights):.2f}")
print(f"  Mean    : {all_heights.mean():.2f}")
print(f"  Maximum : {all_heights.max():.2f}")

print("\nObject-size distribution")

print(
    f"  Small   (<1% image) : "
    f"{small.sum():4d} "
    f"({small.mean() * 100:.2f}%)"
)

print(
    f"  Medium  (1-5%)      : "
    f"{medium.sum():4d} "
    f"({medium.mean() * 100:.2f}%)"
)

print(
    f"  Large   (>=5%)      : "
    f"{large.sum():4d} "
    f"({large.mean() * 100:.2f}%)"
)

print("\nAnalysis complete.")