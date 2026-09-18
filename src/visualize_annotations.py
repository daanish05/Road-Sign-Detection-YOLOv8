from pathlib import Path
import random
import yaml
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

IMAGES_DIR = DATASET_DIR / "train" / "images"
LABELS_DIR = DATASET_DIR / "train" / "labels"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 12


# ============================================================
# LOAD CLASS NAMES
# ============================================================

yaml_path = DATASET_DIR / "data.yaml"

with open(yaml_path, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

class_names = data["names"]

print(f"Number of classes: {len(class_names)}")


# ============================================================
# FIND IMAGES
# ============================================================

image_files = [
    file for file in IMAGES_DIR.iterdir()
    if file.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
]

print(f"Training images found: {len(image_files)}")


# ============================================================
# SELECT RANDOM IMAGES
# ============================================================

random.seed(42)

selected_images = random.sample(
    image_files,
    min(NUM_IMAGES, len(image_files))
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    4,
    figsize=(20, 15)
)

axes = axes.flatten()


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for ax, image_path in zip(axes, selected_images):

    label_path = LABELS_DIR / f"{image_path.stem}.txt"

    image = Image.open(image_path).convert("RGB")

    width, height = image.size

    draw = ImageDraw.Draw(image)

    # --------------------------------------------------------
    # Read YOLO annotation
    # --------------------------------------------------------

    if label_path.exists():

        with open(label_path, "r", encoding="utf-8") as file:

            lines = file.readlines()

        for line in lines:

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

            # ------------------------------------------------
            # Convert YOLO normalized coordinates to pixels
            # ------------------------------------------------

            x_center *= width
            y_center *= height

            box_width *= width
            box_height *= height

            x1 = int(x_center - box_width / 2)
            y1 = int(y_center - box_height / 2)

            x2 = int(x_center + box_width / 2)
            y2 = int(y_center + box_height / 2)

            # ------------------------------------------------
            # Keep coordinates inside image
            # ------------------------------------------------

            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))

            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # ------------------------------------------------
            # Class name
            # ------------------------------------------------

            if class_id < len(class_names):
                class_name = class_names[class_id]
            else:
                class_name = "Unknown"

            label_text = f"{class_id}: {class_name}"

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            draw.rectangle(
                [x1, y1, x2, y2],
                outline="red",
                width=3
            )

            # ------------------------------------------------
            # Draw label background
            # ------------------------------------------------

            try:
                bbox = draw.textbbox(
                    (x1, y1),
                    label_text
                )

                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]

            except AttributeError:

                text_width, text_height = draw.textsize(
                    label_text
                )

            label_y = max(0, y1 - text_height - 4)

            draw.rectangle(
                [
                    x1,
                    label_y,
                    x1 + text_width + 6,
                    label_y + text_height + 4
                ],
                fill="red"
            )

            draw.text(
                (x1 + 3, label_y + 2),
                label_text,
                fill="white"
            )

    # --------------------------------------------------------
    # Display image
    # --------------------------------------------------------

    ax.imshow(image)

    ax.set_title(image_path.name)

    ax.axis("off")


# ============================================================
# SAVE RESULT
# ============================================================

plt.tight_layout()

output_path = OUTPUT_DIR / "annotation_visualization.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("\nAnnotation visualization saved to:")
print(output_path)