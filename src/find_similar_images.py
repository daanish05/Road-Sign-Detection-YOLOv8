from pathlib import Path
from PIL import Image
import imagehash
from collections import defaultdict


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = Path("dataset")

SPLITS = ["train", "valid", "test"]

HASH_SIZE = 16

# Maximum perceptual-hash distance to consider two images
# potentially similar.
MAX_DISTANCE = 8


# ============================================================
# COLLECT IMAGES
# ============================================================

images = []

for split in SPLITS:

    images_dir = DATASET_DIR / split / "images"

    if not images_dir.exists():
        continue

    for image_path in images_dir.iterdir():

        if image_path.suffix.lower() not in [
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        ]:
            continue

        images.append(
            {
                "split": split,
                "path": image_path
            }
        )


print("=" * 70)
print("SIMILAR IMAGE ANALYSIS")
print("=" * 70)

print(f"\nTotal images: {len(images)}")


# ============================================================
# CALCULATE PERCEPTUAL HASHES
# ============================================================

hashes = []

for index, item in enumerate(images):

    try:

        image = Image.open(
            item["path"]
        ).convert("RGB")

        phash = imagehash.phash(
            image,
            hash_size=HASH_SIZE
        )

        hashes.append(
            {
                "split": item["split"],
                "path": item["path"],
                "hash": phash
            }
        )

    except Exception as error:

        print(
            f"Could not process "
            f"{item['path'].name}: {error}"
        )


print(
    f"Successfully hashed: {len(hashes)}"
)


# ============================================================
# FIND SIMILAR IMAGES
# ============================================================

similar_pairs = []

for i in range(len(hashes)):

    for j in range(i + 1, len(hashes)):

        distance = (
            hashes[i]["hash"] -
            hashes[j]["hash"]
        )

        if distance <= MAX_DISTANCE:

            similar_pairs.append(
                {
                    "image1": hashes[i],
                    "image2": hashes[j],
                    "distance": distance
                }
            )


# ============================================================
# SORT
# ============================================================

similar_pairs.sort(
    key=lambda x: x["distance"]
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("SIMILAR IMAGE PAIRS")
print("=" * 70)

print(
    f"\nPairs with perceptual distance "
    f"<= {MAX_DISTANCE}: "
    f"{len(similar_pairs)}"
)


# ============================================================
# SHOW FIRST 100
# ============================================================

for pair in similar_pairs[:100]:

    image1 = pair["image1"]
    image2 = pair["image2"]

    print(
        f"\nDistance: {pair['distance']}"
    )

    print(
        f"  {image1['split']:5} | "
        f"{image1['path'].name}"
    )

    print(
        f"  {image2['split']:5} | "
        f"{image2['path'].name}"
    )


# ============================================================
# CROSS-SPLIT LEAKAGE
# ============================================================

cross_split_pairs = []

for pair in similar_pairs:

    split1 = pair["image1"]["split"]
    split2 = pair["image2"]["split"]

    if split1 != split2:

        cross_split_pairs.append(pair)


print("\n" + "=" * 70)
print("POTENTIAL CROSS-SPLIT LEAKAGE")
print("=" * 70)

print(
    f"\nSimilar pairs occurring across "
    f"different splits: "
    f"{len(cross_split_pairs)}"
)


for pair in cross_split_pairs[:100]:

    image1 = pair["image1"]
    image2 = pair["image2"]

    print(
        f"\nDistance: {pair['distance']}"
    )

    print(
        f"  {image1['split']:5} | "
        f"{image1['path'].name}"
    )

    print(
        f"  {image2['split']:5} | "
        f"{image2['path'].name}"
    )


# ============================================================
# SAVE REPORT
# ============================================================

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

report_path = (
    results_dir /
    "similar_images_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "SIMILAR IMAGE ANALYSIS\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    report.write(
        f"Total images: {len(images)}\n"
    )

    report.write(
        f"Hashed images: {len(hashes)}\n"
    )

    report.write(
        f"Hash threshold: {MAX_DISTANCE}\n"
    )

    report.write(
        f"Similar pairs: {len(similar_pairs)}\n"
    )

    report.write(
        f"Cross-split similar pairs: "
        f"{len(cross_split_pairs)}\n\n"
    )

    report.write(
        "=" * 70 + "\n"
    )

    report.write(
        "CROSS-SPLIT PAIRS\n"
    )

    report.write(
        "=" * 70 + "\n\n"
    )

    for pair in cross_split_pairs:

        image1 = pair["image1"]
        image2 = pair["image2"]

        report.write(
            f"Distance: {pair['distance']}\n"
        )

        report.write(
            f"{image1['split']} | "
            f"{image1['path'].name}\n"
        )

        report.write(
            f"{image2['split']} | "
            f"{image2['path'].name}\n\n"
        )


print(
    f"\nReport saved to: {report_path}"
)