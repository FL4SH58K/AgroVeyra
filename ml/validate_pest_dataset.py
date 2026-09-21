"""Corrupt-file check for the IP102 pest dataset (classification layout).

Mirrors ml/validate_dataset.py, which was used on the disease dataset: PIL verify() over
every image, split by split, so a bad download is caught before any GPU time is spent.
Also reports the class-directory count and any empty class dirs, because IP102's 102
numeric folders (0..101) must all be non-empty for a 102-class classifier to make sense.

Usage:  ml\\.venv\\Scripts\\python.exe ml\\validate_pest_dataset.py
"""

import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent / "data" / "pest" / "ip102" / "classification"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def check_split(split_dir: Path):
    total = 0
    bad = []
    per_class = {}
    for class_dir in sorted(split_dir.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else 0):
        if not class_dir.is_dir():
            continue
        count = 0
        for img_path in class_dir.iterdir():
            if img_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            total += 1
            count += 1
            try:
                with Image.open(img_path) as image:
                    image.verify()
            except Exception as error:
                bad.append((str(img_path), str(error)))
        per_class[class_dir.name] = count
    return total, bad, per_class


def main():
    if not ROOT.is_dir():
        print(f"missing dataset root: {ROOT}")
        return 1

    grand_total = 0
    grand_bad = []
    for split in ("train", "val", "test"):
        split_dir = ROOT / split
        if not split_dir.is_dir():
            print(f"{split}: MISSING ({split_dir})", flush=True)
            continue
        started = time.time()
        total, bad, per_class = check_split(split_dir)
        grand_total += total
        grand_bad += bad
        empty = [name for name, count in per_class.items() if count == 0]
        print(
            f"{split}: {total} images checked, {len(bad)} corrupt, "
            f"{len(per_class)} class dirs, {len(empty)} empty, {time.time() - started:.0f}s",
            flush=True,
        )
        for path, error in bad[:10]:
            print(f"   CORRUPT {path}: {error}", flush=True)
        if empty:
            print(f"   empty class dirs: {empty[:10]}", flush=True)

    print(f"TOTAL: {grand_total} images, {len(grand_bad)} corrupt", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
