from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

CLASS_ID = 203
CLASS_NAME = "Pedestrian crossing"

DATASET = Path("dataset")
OUTPUT = Path("results/class_203_contact_sheet.jpg")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

examples = []

label_dir = DATASET / "train" / "labels"
image_dir = DATASET / "train" / "images"

for label_path in sorted(label_dir.glob("*.txt")):

    has_class = False

    for line in label_path.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        if int(parts[0]) == CLASS_ID:
            has_class = True
            break

    if not has_class:
        continue

    for ext in IMAGE_EXTENSIONS:

        image_path = image_dir / (label_path.stem + ext)

        if image_path.exists():
            examples.append(image_path)
            break


print(f"Found {len(examples)} unique class-{CLASS_ID} images.")


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

THUMB_WIDTH = 250
THUMB_HEIGHT = 200

COLUMNS = 4
ROWS = (len(examples) + COLUMNS - 1) // COLUMNS

LABEL_HEIGHT = 35

sheet_width = COLUMNS * THUMB_WIDTH
sheet_height = ROWS * (THUMB_HEIGHT + LABEL_HEIGHT)

sheet = Image.new(
    "RGB",
    (sheet_width, sheet_height),
    "white"
)

draw = ImageDraw.Draw(sheet)


for index, image_path in enumerate(examples):

    image = Image.open(image_path).convert("RGB")

    image.thumbnail(
        (THUMB_WIDTH - 10, THUMB_HEIGHT - 10)
    )

    x = (index % COLUMNS) * THUMB_WIDTH
    y = (index // COLUMNS) * (THUMB_HEIGHT + LABEL_HEIGHT)

    image_x = x + (THUMB_WIDTH - image.width) // 2
    image_y = y + (THUMB_HEIGHT - image.height) // 2

    sheet.paste(
        image,
        (image_x, image_y)
    )

    label = f"{index + 1}: {image_path.name}"

    # Keep label readable
    if len(label) > 35:
        label = label[:32] + "..."

    draw.text(
        (x + 5, y + THUMB_HEIGHT + 5),
        label,
        fill="black"
    )


OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

sheet.save(
    OUTPUT,
    quality=95
)

print(f"\nContact sheet saved to:")
print(OUTPUT)