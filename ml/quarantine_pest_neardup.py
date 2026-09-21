"""Apply the near-duplicate quarantine to the scoped pest dataset.

Reads ml/_pest_quarantine_list.csv (built by ml/pest_quarantine_plan.py) and moves the listed
files out of the training/eval splits into ml/data/pest/quarantine/neardup_<split>/<slug>/.

Dry run by default - it prints what it would do and changes nothing. Pass --apply to move.
Files are MOVED, never deleted, which is the same pattern used for the disease dataset's
mislabeled_wheat_healthy quarantine: the evidence stays on disk and the move is reversible.

Usage:
    ml\\.venv\\Scripts\\python.exe ml\\quarantine_pest_neardup.py            # dry run
    ml\\.venv\\Scripts\\python.exe ml\\quarantine_pest_neardup.py --apply    # move
"""

import csv
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DST = ROOT / "data" / "pest" / "ip102_scoped"
QUARANTINE = ROOT / "data" / "pest" / "quarantine"
LIST = ROOT / "_pest_quarantine_list.csv"
SPLITS = ("train", "val", "test")


def count_split(split: str) -> int:
    split_dir = DST / split
    if not split_dir.is_dir():
        return 0
    return sum(1 for class_dir in split_dir.iterdir() if class_dir.is_dir()
               for _ in class_dir.iterdir())


def main(argv) -> int:
    apply = "--apply" in argv
    if not LIST.is_file():
        print(f"missing {LIST} - run ml/pest_quarantine_plan.py first")
        return 1

    rows = list(csv.DictReader(LIST.open(encoding="utf-8")))
    before = {split: count_split(split) for split in SPLITS}

    planned = defaultdict(int)
    missing = []
    for row in rows:
        source = DST / row["split"] / row["slug"] / row["filename"]
        if not source.is_file():
            missing.append(str(source))
            continue
        planned[row["split"]] += 1
        if apply:
            destination_dir = QUARANTINE / f"neardup_{row['split']}" / row["slug"]
            destination_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination_dir / row["filename"]))

    verb = "MOVED" if apply else "would move"
    print(f"{verb}: " + ", ".join(f"{split}={planned[split]}" for split in SPLITS))
    if missing:
        print(f"missing files (already moved?): {len(missing)}")
        for path in missing[:5]:
            print(f"  {path}")

    if apply:
        after = {split: count_split(split) for split in SPLITS}
        for split in SPLITS:
            print(f"  {split}: {before[split]} -> {after[split]}")
        print(f"quarantined under {QUARANTINE}")
    else:
        print("dry run - nothing changed. Re-run with --apply to move the files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
