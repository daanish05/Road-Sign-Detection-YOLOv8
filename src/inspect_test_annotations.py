from pathlib import Path

DATASET_DIR = Path("dataset")
TEST_LABELS = DATASET_DIR / "test" / "labels"
TEST_IMAGES = DATASET_DIR / "test" / "images"

print("=" * 80)
print("TEST SET ANNOTATION INSPECTION")
print("=" * 80)

label_files = sorted(TEST_LABELS.glob("*.txt"))

print(f"\nTest label files: {len(label_files)}")

for label_file in label_files:

    print("\n" + "-" * 80)
    print(f"Label file: {label_file.name}")

    image_stem = label_file.stem

    # Find corresponding image
    image_files = []

    for extension in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
        candidate = TEST_IMAGES / (image_stem + extension)

        if candidate.exists():
            image_files.append(candidate)

    if image_files:
        print(f"Image: {image_files[0].name}")
    else:
        print("Image: NOT FOUND")

    with open(label_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    print(f"Annotations: {len(lines)}")

    for i, line in enumerate(lines, start=1):

        parts = line.split()

        if len(parts) != 5:
            print(f"  Annotation {i}: INVALID -> {line}")
            continue

        class_id, x, y, w, h = map(float, parts)

        print(f"  Annotation {i}:")
        print(f"    Class ID : {int(class_id)}")
        print(f"    x_center : {x:.6f}")
        print(f"    y_center : {y:.6f}")
        print(f"    width    : {w:.6f}")
        print(f"    height   : {h:.6f}")
        print(f"    area     : {w * h * 100:.2f}% of image")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)