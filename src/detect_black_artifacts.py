from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Pixels darker than this are considered black
BLACK_THRESHOLD = 25

# Minimum suspicious component size
MIN_AREA = 100

# Maximum component size
# We don't want to classify a huge naturally dark region
# as a small black patch.
MAX_AREA = 5000

# Minimum width/height
MIN_WIDTH = 8
MIN_HEIGHT = 8

# How rectangular/solid the region should be
MIN_SOLIDITY = 0.70


# ============================================================
# DETECT BLACK RECTANGULAR COMPONENTS
# ============================================================

def detect_black_regions(image_path):

    image = cv2.imread(str(image_path))

    if image is None:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold dark pixels
    black_mask = cv2.inRange(
        gray,
        0,
        BLACK_THRESHOLD
    )

    # Small morphological cleanup
    kernel = np.ones((3, 3), np.uint8)

    black_mask = cv2.morphologyEx(
        black_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    # Find connected components
    contours, _ = cv2.findContours(
        black_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    suspicious_regions = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        if area > MAX_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width < MIN_WIDTH or height < MIN_HEIGHT:
            continue

        rectangle_area = width * height

        if rectangle_area == 0:
            continue

        # How much of the bounding rectangle is actually occupied
        fill_ratio = area / rectangle_area

        # Solidity = contour area / convex hull area
        hull = cv2.convexHull(contour)

        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = area / hull_area

        # We want regions that are reasonably rectangular/solid
        if fill_ratio >= 0.55 and solidity >= MIN_SOLIDITY:

            suspicious_regions.append({
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "area": area,
                "fill_ratio": fill_ratio,
                "solidity": solidity
            })

    return suspicious_regions


# ============================================================
# ANALYZE DATASET
# ============================================================

all_suspicious_images = []

total_images = 0


for split in SPLITS:

    images_dir = DATASET_DIR / split / "images"

    if not images_dir.exists():
        print(f"Directory not found: {images_dir}")
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

    print(f"\nChecking {split}: {len(image_files)} images")

    for image_path in image_files:

        total_images += 1

        regions = detect_black_regions(image_path)

        if regions:

            all_suspicious_images.append({
                "split": split,
                "path": image_path,
                "regions": regions
            })


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("IMPROVED BLACK ARTIFACT AUDIT")
print("=" * 60)

print(f"\nTotal images checked: {total_images}")

print(
    f"Images containing suspicious black regions: "
    f"{len(all_suspicious_images)}"
)

print(
    f"Percentage: "
    f"{len(all_suspicious_images) / total_images * 100:.2f}%"
)


# ============================================================
# PRINT EXAMPLES
# ============================================================

print("\nTop suspicious images:")

for item in all_suspicious_images[:30]:

    print(
        f"\n{item['split']} | "
        f"{item['path'].name}"
    )

    for region in item["regions"][:10]:

        print(
            f"  region: "
            f"x={region['x']} "
            f"y={region['y']} "
            f"w={region['width']} "
            f"h={region['height']} "
            f"area={region['area']:.0f}"
        )


# ============================================================
# CREATE VISUALIZATION
# ============================================================

sample_count = min(
    12,
    len(all_suspicious_images)
)

if sample_count > 0:

    fig, axes = plt.subplots(
        3,
        4,
        figsize=(18, 14)
    )

    axes = axes.flatten()

    for ax, item in zip(
        axes,
        all_suspicious_images[:sample_count]
    ):

        image = cv2.imread(
            str(item["path"])
        )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # Draw suspicious regions
        for region in item["regions"]:

            x = region["x"]
            y = region["y"]
            w = region["width"]
            h = region["height"]

            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                3
            )

        ax.imshow(image)

        ax.set_title(
            item["path"].name,
            fontsize=7
        )

        ax.axis("off")

    # Hide unused axes
    for ax in axes[sample_count:]:
        ax.axis("off")

    plt.tight_layout()

    output_path = (
        RESULTS_DIR /
        "suspicious_black_regions.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        f"\nVisualization saved to:"
        f"\n{output_path}"
    )

else:

    print(
        "\nNo suspicious rectangular black regions "
        "were detected."
    )