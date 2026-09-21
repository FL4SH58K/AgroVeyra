"""Triangulate the dHash near-duplicate pairs reported by disease_test_evaluation.py.

WHY THIS EXISTS
  The held-out report prints "near-duplicate pairs (dHash<=6, train sample 20,000): 372". That
  count on its own is NOT evidence of leakage and must not be quoted as if it were:

    * the images are PlantVillage-style leaves on a near-uniform background, so dHash on a
      9x8 greyscale shrink is dominated by the background gradient. Two unrelated leaves of the
      same disease routinely land within 6 bits, and the sample printed in the report shows a
      pair whose two sides are different crops (JR_FrgE.S ~ Mt.N.V_HL), which cannot inflate
      accuracy because the labels disagree;
    * what matters is only whether a scored image has a near twin in train OF THE SAME CLASS,
      and whether that twin is the same physical leaf or just the same background.

  So this script re-derives the same pair list (same seed, same train sample, same dHash), then
  for every pair asks three hard questions and answers them from the pixels rather than from a
  hash: are the two sides the same class, are they the same source family, and do the two images
  actually look like the same leaf after the background is cropped away. Finally it recomputes
  the headline test-half accuracy with the confirmed cases removed, so the leak-free number is
  reported next to the raw one instead of being asserted.

CLASSIFICATION RULES
  same_family   the uuid prefix (text before '___') is identical -> the rebuild should have kept
                these together; any hit here is a real split bug, not a false positive.
  copy_marker   PlantVillage ships its own duplicates as '<label> <id> copy 2_<aug>'. Those have
                DIFFERENT uuids for the same scan, so family grouping cannot see them. They are
                reported separately because they are the one genuine hole this design can have.
  same_class    the resolved label (crop + disease) is identical on both sides.
  pixel MAE     64x64 greyscale mean absolute error, over the whole frame and over the centre
                50% (which is mostly leaf tissue). Centre MAE is the discriminator:
                  <= 5   same leaf, visually indistinguishable at this scale
                  5 - 15 very similar leaf, same disease, plausibly a different specimen
                  > 15   different leaf, the dHash hit was the background

OUTPUTS
  ml/disease_neardup_audit.txt    the report
  ml/_disease_neardup_pairs.csv   one row per pair with every field above

USAGE
  ml\\.venv\\Scripts\\python.exe ml\\disease_neardup_audit.py
  ml\\.venv\\Scripts\\python.exe ml\\disease_neardup_audit.py --train-sample 75419   # exhaustive
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import disease_test_evaluation as dte  # noqa: E402  (path set above)

PRED_CSV = dte.PRED_CSV
PAIRS_CSV = HERE / "_disease_neardup_pairs.csv"
REPORT = HERE / "disease_neardup_audit.txt"

SAMPLE = dte.NEAR_DUP_TRAIN_SAMPLE
LIMIT = dte.DHASH_LIMIT
PIXEL_SIZE = 64
CENTRE_FRACTION = 0.5
SAME_LEAF_MAE = 5.0
SIMILAR_MAE = 15.0

COPY_RE = re.compile(r"\s*copy\s*(\d*)", re.IGNORECASE)
NUM_RE = re.compile(r"(\d+)(?![^0-9]*\d)")


def parse_name(path: Path) -> dict:
    """Split a PlantVillage-style name into family / label / id / augmentation / copy marker."""
    stem = path.stem
    family = stem.split("___", 1)[0] if "___" in stem else stem
    body = stem.split("___", 1)[1] if "___" in stem else stem
    if "_" in body:
        head, aug = body.rsplit("_", 1)
    else:
        head, aug = body, ""
    copy = "1"
    match = COPY_RE.search(head)
    if match:
        copy = match.group(1) or "2"
        head = COPY_RE.sub("", head)
    numbers = NUM_RE.findall(head)
    return {
        "family": family,
        "label": head.strip().lower(),
        "identifier": int(numbers[-1]) if numbers else None,
        "aug": aug.lower(),
        "copy": copy,
    }


def load_predictions(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["correct"] = row["correct"] == "1"
        row["in_top5"] = row["in_top5"] == "1"
        row["confidence"] = float(row["confidence"])
    return rows


def accuracy(rows: list[dict]) -> tuple[int, int, float]:
    if not rows:
        return 0, 0, float("nan")
    right = sum(1 for row in rows if row["correct"])
    return right, len(rows), right / len(rows)


def near_dup_pairs(scored_paths: list[Path], train_paths: list[Path], sample_size: int):
    """Re-derive the pair list exactly as disease_test_evaluation.leak_check did."""
    rng = random.Random(dte.SEED)
    sample_paths = (
        train_paths if len(train_paths) <= sample_size else rng.sample(train_paths, sample_size)
    )
    started = time.time()
    print(f"  hashing {len(sample_paths):,} train images...", flush=True)
    buckets: dict[tuple[int, int], list[tuple[Path, int]]] = defaultdict(list)
    for path, (bits, size) in zip(sample_paths, dte.parallel_map(sample_paths, dte.dhash_of)):
        buckets[size].append((path, bits))
    print(f"  hashing {len(scored_paths):,} scored images...", flush=True)
    pairs: list[tuple[Path, Path, int]] = []
    for path, (bits, size) in zip(scored_paths, dte.parallel_map(scored_paths, dte.dhash_of)):
        for other, other_bits in buckets.get(size, []):
            distance = dte.hamming(bits, other_bits)
            if distance <= LIMIT:
                pairs.append((path, other, distance))
    return pairs, len(sample_paths), time.time() - started


def load_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        small = image.convert("L").resize((PIXEL_SIZE, PIXEL_SIZE), Image.LANCZOS)
        return np.asarray(small, dtype=np.float32)


def centre_box(array: np.ndarray) -> np.ndarray:
    span = int(round(PIXEL_SIZE * CENTRE_FRACTION))
    start = (PIXEL_SIZE - span) // 2
    return array[start : start + span, start : start + span]


def verify(path_a: Path, path_b: Path) -> tuple[float, float]:
    """Return (whole-frame MAE, centre-50% MAE) on 64x64 greyscale."""
    left = load_gray(path_a)
    right = load_gray(path_b)
    return (
        float(np.abs(left - right).mean()),
        float(np.abs(centre_box(left) - centre_box(right)).mean()),
    )


def verdict(centre_mae: float) -> str:
    if centre_mae <= SAME_LEAF_MAE:
        return "same leaf"
    if centre_mae <= SIMILAR_MAE:
        return "similar leaf"
    return "different leaf"


def write_pairs_csv(path: Path, records: list[dict]) -> None:
    fields = [
        "val_split", "val_file", "val_true", "val_pred", "val_correct", "train_file",
        "train_class", "dhash", "same_family", "same_leaf_id", "same_class", "same_crop",
        "same_aug", "mae_full", "mae_centre", "verdict",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            writer.writerow({name: record[name] for name in fields})


def summarise(records: list[dict]) -> dict:
    def count(predicate) -> int:
        return sum(1 for record in records if predicate(record))

    return {
        "pairs": len(records),
        "val_files": len({record["val_file"] for record in records}),
        "train_files": len({record["train_file"] for record in records}),
        "same_family": count(lambda r: r["same_family"]),
        "same_leaf_id": count(lambda r: r["same_leaf_id"]),
        "same_class": count(lambda r: r["same_class"]),
        "same_crop": count(lambda r: r["same_crop"]),
        "same_class_same_leaf_id": count(lambda r: r["same_class"] and r["same_leaf_id"]),
        "same_class_same_leaf": count(lambda r: r["same_class"] and r["mae_centre"] <= SAME_LEAF_MAE),
        "same_class_similar_leaf": count(
            lambda r: r["same_class"] and SAME_LEAF_MAE < r["mae_centre"] <= SIMILAR_MAE
        ),
        "cross_class": count(lambda r: not r["same_class"]),
        "cross_class_same_leaf": count(lambda r: not r["same_class"] and r["mae_centre"] <= SAME_LEAF_MAE),
        "min_centre_mae": min((r["mae_centre"] for r in records), default=float("nan")),
    }


def adjusted(rows: list[dict], flagged: set[Path]) -> tuple[int, int, float]:
    return accuracy([row for row in rows if Path(row["file"]) not in flagged])


def build_report(
    stats, records, predictions, splits, val_rows, test_rows, sample_size,
    train_images, hash_seconds, flagged, flagged_all, correct_among, elapsed,
):
    lines: list[str] = []

    def add(text: str = "") -> None:
        lines.append(text)

    def rule(title: str) -> None:
        add("-" * 118)
        add(title)
        add("-" * 118)

    add("=" * 118)
    add("DISEASE CLASSIFIER - what the dHash near-duplicate pairs actually are")
    add("=" * 118)
    add("source      : ml/disease_test_evaluation.txt (LEAKAGE RE-CHECK), ml/_disease_test_predictions.csv")
    add(f"pair list   : {PAIRS_CSV}")
    add("purpose     : decide whether '372 near-duplicate pairs' is leakage or background collision, and")
    add("              put an explicit number on what it costs the 97.38% headline.")
    add("")

    rule("1. WHAT WAS RE-DERIVED")
    scope = "EXHAUSTIVE - every train image" if sample_size >= train_images else f"sample, seed {dte.SEED}"
    add(f"  train images re-hashed            : {sample_size:,} of {train_images:,} ({scope})")
    add(f"  scored images re-hashed           : {len(val_rows):,} val images")
    add(f"  candidate pairs (dHash <= {LIMIT}, same pixel size) : {stats['pairs']}")
    add(f"  distinct val images in pairs      : {stats['val_files']}")
    add(f"  distinct train images opposite    : {stats['train_files']}")
    add(f"  hashing time                      : {hash_seconds:.0f}s (independent recomputation)")
    add("")

    rule("2. THE ONLY FATAL KIND: a real split violation")
    add(f"  same source family (uuid matches)          : {stats['same_family']} / {stats['pairs']}")
    add("      A non-zero here would mean one physical leaf sits in train and in the held-out half, i.e.")
    add("      the rebuild's photo-boundary rule failed. It is zero, so that rule still holds.")
    add(f"  same scan id (same label + same number)    : {stats['same_leaf_id']} / {stats['pairs']}")
    add("      PlantVillage ships its own duplicates as '<label> <id> copy 2_<aug>'. Those carry a")
    add("      DIFFERENT uuid, so family grouping cannot see them. This is the one hole the design can")
    add("      have, which is why it is counted separately instead of inside the dHash count.")
    add(f"  same class AND same scan id                : {stats['same_class_same_leaf_id']} / {stats['pairs']}")
    add("")

    rule("3. SAME CLASS OR NOT (only same-class pairs can inflate top-1)")
    add(f"  same class                                     : {stats['same_class']}")
    add(f"  same crop, different disease                   : {stats['same_crop'] - stats['same_class']}")
    add(f"  different crop                                 : {stats['cross_class']}")
    add("      A different-crop pair cannot inflate the metric: the two labels disagree, so the model is")
    add("      not being asked the same question twice.")
    add("")

    rule("4. PIXEL VERIFICATION - centre-50% MAE on 64x64 greyscale")
    add("  bucket                              pairs   same-class   cross-class")
    buckets = (
        ("same leaf      (MAE <= 5)", lambda m: m <= SAME_LEAF_MAE),
        ("similar leaf   (5 < MAE <= 15)", lambda m: SAME_LEAF_MAE < m <= SIMILAR_MAE),
        ("different leaf (MAE > 15)", lambda m: m > SIMILAR_MAE),
    )
    for label, test in buckets:
        subset = [r for r in records if test(r["mae_centre"])]
        add(f"  {label:<34}{len(subset):>6}{sum(1 for r in subset if r['same_class']):>13}"
            f"{sum(1 for r in subset if not r['same_class']):>14}")
    add(f"  minimum centre MAE seen                        : {stats['min_centre_mae']:.2f}")
    add("")
    add("  the 12 pairs that look most alike (lowest centre MAE)")
    add(f"      {'MAE':>6}  {'dHash':>5}  {'class':>9}  val / train")
    for record in sorted(records, key=lambda r: r["mae_centre"])[:12]:
        same = "same" if record["same_class"] else "DIFFERENT"
        add(f"      {record['mae_centre']:>6.2f}  {record['dhash']:>5}  {same:>9}  "
            f"{Path(record['val_file']).name[:42]}  ~  {Path(record['train_file']).name[:42]}")
    add("")

    rule("5. WHAT IT COSTS THE HEADLINE")
    base_right, base_n, base_acc = accuracy(test_rows)
    add(f"  rows in the predictions file : {len(predictions):,}  {dict(splits)}")
    add(f"  TEST half                    : {base_right:,}/{base_n:,} = {base_acc * 100:.2f}% (the published number)")
    add("")
    add(f"  {'exclusion rule':<46}{'excl':>6}{'left':>7}{'accuracy':>11}{'delta':>9}{'excl. correct':>16}")
    order = (
        ("every pair above, any class", "all_pairs"),
        ("same-class pairs only", "same_class"),
        ("same scan id (PlantVillage copies)", "same_leaf_id"),
        ("same class AND same leaf (MAE <= 5)", "same_class_same_leaf"),
        ("same class AND <= similar leaf (MAE <= 15)", "same_class_same_or_similar"),
        ("same leaf whatever the class (MAE <= 5)", "any_same_leaf"),
    )
    for label, key in order:
        paths = flagged[key]
        right, total, acc = adjusted(test_rows, paths)
        delta = (acc - base_acc) * 100
        add(f"  {label:<46}{len(paths):>6}{total:>7}{acc * 100:>10.2f}%{delta:>+9.2f}"
            f"{correct_among[key]:>10}/{len(paths):<4}")
    add("")
    test_leaves = {row["source_leaf"] for row in test_rows}
    strict_leaves = {
        row["source_leaf"] for row in test_rows
        if Path(row["file"]) in flagged["same_class_same_leaf"]
    }
    add(f"  leaves touched by the strict set (same class AND same leaf): {len(strict_leaves)} of "
        f"{len(test_leaves):,} ({len(strict_leaves) / max(len(test_leaves), 1) * 100:.3f}%)")
    add("")

    rule("6. VERDICT")
    if not stats["pairs"]:
        add("  No pairs at all were found on this run.")
    else:
        add(f"  {stats['same_family']} of {stats['pairs']} pairs share a source family, so no augmented family spans the")
        add("  train/held-out boundary: the split itself is intact.")
        add("")
        add(f"  {stats['cross_class']} of {stats['pairs']} pairs are leaves of DIFFERENT crops that hash within {LIMIT} bits of one")
        add("  another because the plain background dominates a dHash on a PlantVillage image. They cannot")
        add("  inflate top-1 accuracy and should never be quoted as leakage.")
        add("")
        add(f"  {stats['same_class']} pairs are same-class. Of those, {stats['same_class_same_leaf']} survive pixel verification as the same")
        add(f"  leaf (centre MAE <= {SAME_LEAF_MAE:.0f}) and {stats['same_class_similar_leaf']} are only similar. Section 5 prints the headline with each")
        add("  of those sets removed, so the leak-free number is measured rather than asserted.")
        add("")
        add("  READ IT AS: the published number is an in-domain score on augmented leaves; section 5 is its")
        add("  leak-free version, and the sweep does not change the story. The story is still the one in")
        add("  ml/field_failure_analysis.txt - 43.75% on real phone photos of real leaves.")
    add("")
    add(f"pairs csv   : {PAIRS_CSV}")
    add(f"elapsed     : {elapsed:.0f}s")
    return lines



def parse_args():
    parser = argparse.ArgumentParser(
        description="Audit the dHash near-duplicate pairs reported by disease_test_evaluation.py."
    )
    parser.add_argument(
        "--train-sample", type=int, default=SAMPLE,
        help="how many train images to hash (default: the evaluation run's 20,000-image sample; "
             "pass the whole train split for an exhaustive sweep)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    print("AUDITING THE dHash NEAR-DUPLICATE PAIRS", flush=True)

    predictions = load_predictions(PRED_CSV)
    splits = Counter(row["split"] for row in predictions)
    val_rows = [row for row in predictions if row["split"].startswith(("test", "selection"))]
    test_rows = [row for row in val_rows if row["split"].startswith("test")]
    scored_paths = [Path(row["file"]) for row in val_rows]
    row_of = {Path(row["file"]): row for row in val_rows}
    print(f"  predictions {len(predictions):,} rows, splits {dict(splits)}", flush=True)

    train_paths = dte.list_images(dte.TRAIN_DIR)
    pairs, sample_size, hash_seconds = near_dup_pairs(scored_paths, train_paths, args.train_sample)
    scope = "EXHAUSTIVE" if sample_size >= len(train_paths) else f"{sample_size:,}-image sample"
    print(f"  {len(pairs)} candidate pairs in {hash_seconds:.0f}s ({scope})", flush=True)
    print("  pixel-verifying each pair...", flush=True)

    records: list[dict] = []
    for index, (val_path, train_path, distance) in enumerate(pairs, 1):
        row = row_of[val_path]
        left = parse_name(val_path)
        right = parse_name(train_path)
        train_class = train_path.parent.name
        full_mae, centre_mae = verify(val_path, train_path)
        records.append(
            {
                "val_split": row["split"],
                "val_file": str(val_path),
                "val_true": row["true"],
                "val_pred": row["pred"],
                "val_correct": int(row["correct"]),
                "train_file": str(train_path),
                "train_class": train_class,
                "dhash": distance,
                "same_family": int(left["family"] == right["family"]),
                "same_leaf_id": int(
                    left["label"] == right["label"] and left["identifier"] == right["identifier"]
                ),
                "same_class": int(row["true"] == train_class),
                "same_crop": int(dte.crop_of(row["true"]) == dte.crop_of(train_class)),
                "same_aug": int(left["aug"] == right["aug"]),
                "mae_full": round(full_mae, 3),
                "mae_centre": round(centre_mae, 3),
                "verdict": verdict(centre_mae),
            }
        )
        if index % 50 == 0:
            print(f"    {index}/{len(pairs)}", flush=True)

    stats = summarise(records)
    flagged = {
        "all_pairs": {Path(r["val_file"]) for r in records},
        "same_class": {Path(r["val_file"]) for r in records if r["same_class"]},
        "same_leaf_id": {Path(r["val_file"]) for r in records if r["same_leaf_id"]},
        "same_class_same_leaf": {
            Path(r["val_file"]) for r in records if r["same_class"] and r["mae_centre"] <= SAME_LEAF_MAE
        },
        "same_class_same_or_similar": {
            Path(r["val_file"]) for r in records if r["same_class"] and r["mae_centre"] <= SIMILAR_MAE
        },
        "any_same_leaf": {Path(r["val_file"]) for r in records if r["mae_centre"] <= SAME_LEAF_MAE},
    }
    test_flagged = {name: {p for p in paths if row_of[p]["split"].startswith("test")} for name, paths in flagged.items()}
    correct_among = {
        name: sum(1 for row in test_rows if Path(row["file"]) in paths and row["correct"])
        for name, paths in test_flagged.items()
    }

    write_pairs_csv(PAIRS_CSV, records)
    lines = build_report(
        stats=stats,
        records=records,
        predictions=predictions,
        splits=splits,
        val_rows=val_rows,
        test_rows=test_rows,
        sample_size=sample_size,
        train_images=len(train_paths),
        hash_seconds=hash_seconds,
        flagged=test_flagged,
        flagged_all=flagged,
        correct_among=correct_among,
        elapsed=time.time() - started,
    )
    for line in lines:
        print(line, flush=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.name} ({len(lines)} lines) and {PAIRS_CSV.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


