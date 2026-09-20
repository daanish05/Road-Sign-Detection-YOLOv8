from pathlib import Path
from PIL import Image, ImageDraw
import math

CLASS_ID = 203
CLASS_NAME = "Pedestrian crossing"

DATASET = Path("dataset")
OUTPUT = Path("results/class_203_examples")
OUTPUT.mkdir(parents=True, exist_ok=True)

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

examples = []

for label_path in (DATASET / "train" / "labels").glob("*.txt"):

    for line in label_path.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        if class_id != CLASS_ID:
            continue

        image_path = None

        for ext in image_extensions:
            candidate = DATASET / "train" / "images" / (label_path.stem + ext)

            if candidate.exists():
                image_path = candidate
                break

        if image_path:
            examples.append(image_path)
            break


print("=" * 50)
print(f"CLASS {CLASS_ID}: {CLASS_NAME}")
print("=" * 50)
print(f"Training images found: {len(examples)}")

for i, image_path in enumerate(examples, start=1):

    image = Image.open(image_path).convert("RGB")

    # Find the class-203 bounding box
    label_path = DATASET / "train" / "labels" / (image_path.stem + ".txt")

    width, height = image.size

    for line in label_path.read_text().splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id, x, y, w, h = map(float, parts)

        if int(class_id) != CLASS_ID:
            continue

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        draw = ImageDraw.Draw(image)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline="red",
            width=4
        )

        draw.text(
            (x1, max(0, y1 - 20)),
            f"{CLASS_ID} - {CLASS_NAME}",
            fill="red"
        )

        break

    output_path = OUTPUT / f"class203_{i:02d}_{image_path.name}"

    image.save(output_path)

    print(f"{i:02d}: {image_path.name}")


print("\nSaved examples to:")
print(OUTPUT)