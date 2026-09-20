from pathlib import Path

CLASS_ID = 203

for split in ["train", "valid", "test"]:
    label_dir = Path("dataset") / split / "labels"

    total = 0
    files = 0

    for label_file in label_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.strip().split()

            if parts and int(parts[0]) == CLASS_ID:
                total += 1
                files += 1

    print(
        f"{split:5s} | "
        f"annotations: {total:4d} | "
        f"images: {files:4d}"
    )

print("\nClass:", CLASS_ID)
print("Name: Pedestrian crossing")