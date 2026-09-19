from pathlib import Path
from collections import defaultdict

DATASET_DIR = Path("dataset")
SPLITS = ["train", "valid", "test"]


# --------------------------------------------------
# FIND SOURCE GROUP
# --------------------------------------------------

def get_source_stem(filename):
    """
    Example:

    sign19_jpg.rf.abcdef123.jpg
              ^

    becomes:

    sign19_jpg
    """

    name = Path(filename).stem

    if ".rf." in name:
        return name.split(".rf.")[0]

    return name


# --------------------------------------------------
# COLLECT GROUPS
# --------------------------------------------------

groups = defaultdict(lambda: defaultdict(list))

for split in SPLITS:

    image_dir = DATASET_DIR / split / "images"

    for image_path in image_dir.iterdir():

        if not image_path.is_file():
            continue

        source = get_source_stem(image_path.name)

        groups[source][split].append(
            image_path.name
        )


# --------------------------------------------------
# ANALYZE CROSS-SPLIT GROUPS
# --------------------------------------------------

cross_split_groups = []

for source, split_data in groups.items():

    splits_present = [
        split
        for split in SPLITS
        if split in split_data
    ]

    if len(splits_present) > 1:

        cross_split_groups.append(
            (
                source,
                splits_present,
                {
                    split: len(split_data[split])
                    for split in splits_present
                }
            )
        )


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

print("=" * 80)
print("SOURCE GROUP / SPLIT ANALYSIS")
print("=" * 80)

print(f"\nTotal source groups: {len(groups)}")

print(
    f"Groups appearing in multiple splits: "
    f"{len(cross_split_groups)}"
)

if cross_split_groups:

    print("\nCross-split groups:")
    print("-" * 80)

    for source, splits, counts in cross_split_groups[:100]:

        print(
            f"{source:<35} | "
            f"Splits: {', '.join(splits):<20} | "
            f"Counts: {counts}"
        )

else:

    print(
        "\nNo source groups were found across "
        "multiple dataset splits."
    )


# --------------------------------------------------
# GROUP COUNTS BY SPLIT
# --------------------------------------------------

print("\n" + "=" * 80)
print("SOURCE GROUP DISTRIBUTION")
print("=" * 80)

for split in SPLITS:

    split_groups = set()

    image_dir = DATASET_DIR / split / "images"

    for image_path in image_dir.iterdir():

        if image_path.is_file():

            split_groups.add(
                get_source_stem(image_path.name)
            )

    print(
        f"{split:<10}: "
        f"{len(split_groups)} source groups"
    )


print("\nAnalysis complete.")