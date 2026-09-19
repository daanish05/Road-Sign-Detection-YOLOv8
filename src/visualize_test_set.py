from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATASET_DIR = Path("dataset")

IMAGE_DIR = DATASET_DIR / "test" / "images"
LABEL_DIR = DATASET_DIR / "test" / "labels"

OUTPUT = Path("results/test_set_visualization.jpg")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# FIND TEST IMAGES
# --------------------------------------------------

image_files = sorted(
    list(IMAGE_DIR.glob("*.jpg")) +
    list(IMAGE_DIR.glob("*.jpeg")) +
    list(IMAGE_DIR.glob("*.png"))
)

if not image_files:
    print("No test images found.")
    raise SystemExit


font = ImageFont.load_default()

panels = []


# --------------------------------------------------
# PROCESS EACH TEST IMAGE
# --------------------------------------------------

for image_path in image_files:

    image = Image.open(image_path).convert("RGB")

    draw = ImageDraw.Draw(image)

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    labels = []

    if label_path.exists():

        with open(label_path, "r") as f:
            labels = [
                line.strip()
                for line in f
                if line.strip()
            ]

    width, height = image.size

    # ----------------------------------------------
    # DRAW GROUND-TRUTH BOXES
    # ----------------------------------------------

    for line in labels:

        parts = line.split()

        if len(parts) != 5:
            continue

        class_id, x, y, w, h = map(float, parts)

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)

        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        draw.text(
            (10, 10),
            f"Class ID: {int(class_id)}",
            fill="red",
            font=font
        )

    # ----------------------------------------------
    # RESIZE FOR CONTACT SHEET
    # ----------------------------------------------

    image.thumbnail((400, 400))

    panels.append(
        (
            image,
            image_path.name
        )
    )


# --------------------------------------------------
# CREATE CONTACT SHEET
# --------------------------------------------------

columns = 3

rows = (
    len(panels) + columns - 1
) // columns

cell_width = 420
cell_height = 450

sheet = Image.new(
    "RGB",
    (
        columns * cell_width,
        rows * cell_height
    ),
    "white"
)

draw = ImageDraw.Draw(sheet)


# --------------------------------------------------
# PLACE IMAGES
# --------------------------------------------------

for i, (image, filename) in enumerate(panels):

    x = (i % columns) * cell_width
    y = (i // columns) * cell_height

    paste_x = x + (
        cell_width - image.width
    ) // 2

    paste_y = y + 10

    sheet.paste(
        image,
        (paste_x, paste_y)
    )

    draw.text(
        (x + 10, y + 415),
        filename[:55],
        fill="black",
        font=font
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

sheet.save(
    OUTPUT,
    quality=95
)

print(
    f"Saved visualization to: {OUTPUT}"
)