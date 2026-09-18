from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


DATASET_DIR = Path("dataset")
IMAGES_DIR = DATASET_DIR / "train" / "images"

OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

NUM_IMAGES = 12


# Get images
image_files = [
    f for f in IMAGES_DIR.iterdir()
    if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
]

# Use the same fixed seed
import random
random.seed(42)

selected_images = random.sample(
    image_files,
    min(NUM_IMAGES, len(image_files))
)


fig, axes = plt.subplots(
    3,
    4,
    figsize=(20, 15)
)

axes = axes.flatten()


for ax, image_path in zip(axes, selected_images):

    image = Image.open(image_path).convert("RGB")

    ax.imshow(image)
    ax.set_title(image_path.name, fontsize=8)
    ax.axis("off")


plt.tight_layout()

output_path = OUTPUT_DIR / "raw_images.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print(f"Saved raw images to: {output_path}")