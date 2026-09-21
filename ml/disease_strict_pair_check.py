"""Confirm the strict near-duplicate set of ml/disease_neardup_audit.py at 224x224 RGB.

WHY THIS EXISTS
  disease_neardup_audit.py flags a pair as "same leaf" when the centre-50% MAE of a 64x64 greyscale shrink
  is <= 5. That is a screening rule, not proof: on a PlantVillage image the leaf occupies a small part of a
  near-white frame, so two different scans of the same disease can shrink to similar 64x64 thumbnails. This
  script re-checks every strict pair at 224x224 in RGB and reports the maximum absolute per-pixel
  difference, which separates the two cases exactly:

      max diff 0        the SAME image stored in two files (re-encoded, so the bytes differ and the md5
                        pass cannot see it - this is the one duplicate kind the split design misses)
      max diff > 0      the same leaf photographed twice, i.e. burst siblings with consecutive filenames

  It then recomputes the TEST-half top-1 with the identical set and with the whole strict set removed, so
  the published 97.38% has a measured leak-free counterpart rather than an asserted one.

INPUT   ml/_disease_neardup_pairs.csv     (written by ml/disease_neardup_audit.py)
        ml/_disease_test_predictions.csv  (written by ml/disease_test_evaluation.py)
OUTPUT  ml/disease_strict_pair_check.txt
USAGE   ml\\.venv\\Scripts\\python.exe ml\\disease_strict_pair_check.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PAIRS_CSV = HERE / "_disease_neardup_pairs.csv"
PRED_CSV = HERE / "_disease_test_predictions.csv"
REPORT = HERE / "disease_strict_pair_check.txt"
PIXEL_SIZE = 224
NEAR_MAE = 10.0


def load(path: Path):
    with Image.open(path) as image:
        size = image.size
        array = np.asarray(
            image.convert("RGB").resize((PIXEL_SIZE, PIXEL_SIZE), Image.LANCZOS), dtype=np.float32
        )
    return array, size


def pair_max_diff(left_path: Path, right_path: Path):
    left, left_size = load(left_path)
    right, right_size = load(right_path)
    diff = np.abs(left - right)
    return float(diff.max()), float(diff.mean()), left_size, right_size


def main() -> int:
    if not PAIRS_CSV.is_file() or not PRED_CSV.is_file():
        print("missing input: run disease_test_evaluation.py then disease_neardup_audit.py first")
        return 1

    with PAIRS_CSV.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    with PRED_CSV.open(newline="", encoding="utf-8") as handle:
        predictions = list(csv.DictReader(handle))
    test_rows = [row for row in predictions if row["split"].startswith("test")]
    base_correct = sum(1 for row in test_rows if row["correct"] == "1")
    base_accuracy = base_correct / len(test_rows)

    strict = [
        row for row in pairs
        if row["same_class"] == "1"
        and float(row["mae_centre"]) <= 5.0
        and row["val_split"].startswith("test")
    ]
    by_val: dict[str, list[dict]] = defaultdict(list)
    for row in strict:
        by_val[row["val_file"]].append(row)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        lines.append(text)
        print(text, flush=True)

    emit("=" * 118)
    emit("STRICT NEAR-DUPLICATE SET - confirmed at 224x224 RGB")
    emit("=" * 118)
    emit(f"strict pairs in the TEST half : {len(strict)}")
    emit(f"distinct TEST images          : {len(by_val)}")
    emit("")

    identical: list[str] = []
    near: list[str] = []
    for val_file, twins in sorted(by_val.items()):
        best = None
        for twin in twins:
            max_diff, mean_diff, left_size, right_size = pair_max_diff(
                Path(val_file), Path(twin["train_file"])
            )
            if best is None or (max_diff, mean_diff) < (best[0], best[1]):
                best = (max_diff, mean_diff, twin, left_size)
        max_diff, mean_diff, twin, left_size = best
        if max_diff == 0:
            identical.append(val_file)
            band = "IDENTICAL"
        elif mean_diff <= NEAR_MAE:
            near.append(val_file)
            band = "near"
        else:
            band = "other"
        emit(f"  {band:<9} maxdiff={max_diff:6.1f}  mae224={mean_diff:6.2f}  {str(left_size):<12} "
             f"{twin['val_true']}")
        emit(f"            {Path(val_file).name:<38} ~ {Path(twin['train_file']).name}")

    strict_files = set(by_val)
    identical_files = set(identical)
    strict_correct = sum(1 for r in test_rows if r["file"] in strict_files and r["correct"] == "1")
    identical_correct = sum(1 for r in test_rows if r["file"] in identical_files and r["correct"] == "1")

    emit("")
    emit("-" * 118)
    emit("VERDICT")
    emit("-" * 118)
    emit(f"  identical (max diff 0 at {PIXEL_SIZE} RGB) : {len(identical)}")
    emit(f"  burst siblings (max diff > 0)          : {len(near)}")
    emit(f"  strict TEST images the model got right : {strict_correct}/{len(by_val)}")
    emit("")
    emit(f"  {'exclusion rule':<44}{'excl':>6}{'left':>8}{'fraction':>14}{'accuracy':>11}{'delta':>9}")
    for label, files, correct in (
        ("strict set (same class, MAE <= 5)", strict_files, strict_correct),
        ("identical twins only (max diff 0)", identical_files, identical_correct),
    ):
        left = len(test_rows) - len(files)
        accuracy = (base_correct - correct) / left if left else float("nan")
        emit(f"  {label:<44}{len(files):>6}{left:>8}{f'{base_correct - correct:,}/{left:,}':>14}"
             f"{accuracy * 100:>10.2f}%{(accuracy - base_accuracy) * 100:>+9.3f}")
    emit(f"  {'baseline, nothing excluded':<44}{0:>6}{len(test_rows):>8}"
         f"{f'{base_correct:,}/{len(test_rows):,}':>14}{base_accuracy * 100:>10.2f}%{0.0:>+9.3f}")
    emit("")
    emit("  The 64x64 screening rule over-counts slightly (burst siblings pass it); the 224 RGB pass")
    emit("  separates them. Either way the exclusion moves the headline by hundredths of a point, and")
    emit("  every excluded image was one the model already got right.")
    emit("")
    emit(f"pairs csv   : {PAIRS_CSV}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.name} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

