# from pathlib import Path

# CLASS_ID = 203

# for split in ["train", "valid", "test"]:
#     label_dir = Path("dataset") / split / "labels"

#     total = 0
#     files = 0

#     for label_file in label_dir.glob("*.txt"):
#         for line in label_file.read_text().splitlines():
#             parts = line.strip().split()

#             if parts and int(parts[0]) == CLASS_ID:
#                 total += 1
#                 files += 1

#     print(
#         f"{split:5s} | "
#         f"annotations: {total:4d} | "
#         f"images: {files:4d}"
#     )

# print("\nClass:", CLASS_ID)
# print("Name: Pedestrian crossing")






from pathlib import Path

CLASS_ID = 169
CLASS_NAME = "Horn prohibited"

print("=" * 50)
print(f"CLASS {CLASS_ID}: {CLASS_NAME}")
print("=" * 50)

for split in ["train", "valid", "test"]:

    label_dir = Path("dataset") / split / "labels"

    total_annotations = 0
    image_files = set()

    for label_file in label_dir.glob("*.txt"):

        count_in_file = 0

        for line in label_file.read_text().splitlines():

            parts = line.strip().split()

            if len(parts) != 5:
                continue

            if int(parts[0]) == CLASS_ID:
                count_in_file += 1

        if count_in_file > 0:
            total_annotations += count_in_file
            image_files.add(label_file.stem)

    print(
        f"{split:5s} | "
        f"annotations: {total_annotations:4d} | "
        f"images: {len(image_files):4d}"
    )