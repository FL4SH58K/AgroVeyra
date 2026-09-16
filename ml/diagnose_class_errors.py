#!/usr/bin/env python
"""Where does the deployed classifier actually fail, and what should the next training run fix?

`verify_export_parity.py` proves the flatbuffer matches the checkpoint. `benchmark_preprocessing.py`
shows preprocessing is not the bottleneck. This script answers the remaining question: *which
classes* carry the errors, and is the dataset underneath them sound?

Four diagnostics, all on the model's own validation split so the numbers are comparable to
`runs/classify/runs/disease_classifier-3/results.csv`:

  1. DATASET BALANCE      images per class in train/val, flagging classes far below the median
  2. PREDICTION PRIOR     how often each class is *predicted* vs how often it is the *truth*
                          (a ratio >> 1 means the model over-predicts that class)
  3. CONFUSION PAIRS      the off-diagonal cells, worst first: what each wrong answer became
  4. CROP-LEVEL ROLLUP    accuracy with the disease ignored - matters because the app shows the
                          crop first, and a wheat rust mix-up is a far cheaper mistake than
                          calling rice "wheat"

Usage:
    ml/.venv/Scripts/python.exe ml/diagnose_class_errors.py
    ml/.venv/Scripts/python.exe ml/diagnose_class_errors.py --val-per-class 20 --top 30
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_MODEL = PROJECT_ROOT / "backend" / "models" / "agroveyra_model.tflite"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "backend" / "models" / "class_names.json"
DEFAULT_DATA_DIR = HERE / "data" / "merged"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Class-name crops that mean the same plant (the dataset spells them several ways).
FAMILY_ALIASES = {
    "cornmaize": "Corn",
    "cherryincludingsour": "Cherry",
    "pepperbell": "Pepper",
    "pepper": "Pepper",
}


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def load_evaluator():
    sys.path.insert(0, str(HERE))
    import evaluate_real_world as ev

    return ev


def family_of(ev, class_name: str) -> str:
    crop, _ = ev.split_class_name(class_name)
    key = ev.normalize(crop)
    return FAMILY_ALIASES.get(key, crop or "unknown")


def load_val_items(directory: Path, per_class: int) -> list[tuple[Path, str]]:
    """(path, class folder name) pairs, deterministic evenly-spaced first-N per class."""
    items: list[tuple[Path, str]] = []
    for class_dir in sorted(entry for entry in directory.iterdir() if entry.is_dir()):
        files = sorted(file for file in class_dir.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            continue
        step = max(len(files) // per_class, 1)
        for path in files[::step][:per_class]:
            items.append((path, class_dir.name))
    return items


def count_images(directory: Path) -> dict[str, int]:
    """class folder name -> image count (0 for folders that are missing)."""
    counts: dict[str, int] = {}
    if not directory.exists():
        return counts
    for entry in sorted(directory.iterdir()):
        if entry.is_dir():
            counts[entry.name] = sum(1 for file in entry.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)
    return counts


def predict_rows(
    model_path: Path, class_names: list[str], items: list[tuple[Path, str]], center_crop: bool
) -> list[dict]:
    """Top-1 (plus runner-up and margin) for every sampled image, using the deployed pipeline."""
    ev = load_evaluator()
    interpreter_class = ev.import_interpreter()
    interpreter = interpreter_class(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    size = int(input_detail["shape"][1])
    layout = "app" if [int(dim) for dim in input_detail["shape"]][-1] == 3 else "native"

    rows: list[dict] = []
    for path, ground_truth in items:
        pixels = ev.preprocess_image(path, size, center_crop)
        tensor = ev.build_input_tensor(pixels, input_detail, layout)
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        scores = ev.ensure_probabilities(
            ev.dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        )
        order = np.argsort(-scores, kind="stable")
        predicted = class_names[int(order[0])]
        runner_up = class_names[int(order[1])] if len(order) > 1 else ""
        rows.append(
            {
                "image": path.name,
                "ground_truth": ground_truth,
                "predicted": predicted,
                "confidence": float(scores[order[0]]),
                "runner_up": runner_up,
                "runner_up_confidence": float(scores[order[1]]) if len(order) > 1 else 0.0,
                "margin": float(scores[order[0]] - scores[order[1]]) if len(order) > 1 else float(scores[order[0]]),
                "correct": predicted == ground_truth,
            }
        )
    return rows


def report_balance(train_counts: dict[str, int], val_counts: dict[str, int], class_names: list[str], emit) -> None:
    present = [count for count in train_counts.values() if count]
    median = float(np.median(present)) if present else 0.0
    threshold = median * 0.5

    emit("=" * 118)
    emit("1. DATASET BALANCE (images per class; a class under half the median cannot be learned well)")
    emit("=" * 118)
    emit(f"  median train count: {median:.0f}    weak classes (< {threshold:.0f}): "
         f"{sum(1 for name in class_names if train_counts.get(name, 0) < threshold)}/{len(class_names)}")
    emit(f"  {'class':<42} | {'train':>7} | {'val':>6} | {'vs median':>9}")
    for name in sorted(class_names, key=lambda item: (train_counts.get(item, 0), item)):
        train_count = train_counts.get(name, 0)
        ratio = train_count / median if median else 0.0
        flag = "  <-- weak" if train_count < threshold else ""
        emit(f"  {name:<42} | {train_count:>7} | {val_counts.get(name, 0):>6} | {ratio:>8.2f}x{flag}")
    emit("")


def report_prior(rows: list[dict], class_names: list[str], emit, top: int) -> None:
    """ratio = predicted count / true count; a ratio above 1 means the model reaches for that class."""
    truths = Counter(row["ground_truth"] for row in rows)
    predictions = Counter(row["predicted"] for row in rows)
    total = len(rows)

    entries = []
    for name in class_names:
        true_count = truths.get(name, 0)
        pred_count = predictions.get(name, 0)
        if not true_count and not pred_count:
            continue
        ratio = pred_count / true_count if true_count else float("inf")
        entries.append((ratio, name, true_count, pred_count))

    emit("=" * 118)
    emit(f"2. PREDICTION PRIOR vs TRUE PRIOR (top {top}; ratio > 1 = over-predicted)")
    emit("=" * 118)
    emit(f"  {'class':<42} | {'truth':>6} | {'predicted':>9} | {'ratio':>6} | {'share':>6}")
    for ratio, name, true_count, pred_count in sorted(entries, reverse=True)[:top]:
        share = pred_count / total * 100 if total else 0.0
        emit(f"  {name:<42} | {true_count:>6} | {pred_count:>9} | {ratio:>6.2f} | {share:>5.1f}%")
    emit("")
    over = [entry for entry in entries if entry[0] >= 1.25 and entry[3] >= 3]
    if over:
        emit(f"  over-predicted classes (ratio >= 1.25 with at least 3 predictions): {len(over)}")
        emit("    a systematic surplus of one crop family is a prior/label problem, not a bug in the export")
    emit("")


def report_confusions(rows: list[dict], emit, top: int) -> None:
    pairs = Counter(
        (row["ground_truth"], row["predicted"]) for row in rows if not row["correct"]
    )
    mistakes = [row for row in rows if not row["correct"]]

    emit("=" * 118)
    emit(f"3. CONFUSION PAIRS ({len(mistakes)} of {len(rows)} images wrong; top {top})")
    emit("=" * 118)
    emit(f"  {'truth':<38} -> {'predicted':<38} | {'n':>3} | {'mean conf':>9} | {'mean margin':>11}")
    for (truth, predicted), count in pairs.most_common(top):
        subset = [row for row in mistakes if row["ground_truth"] == truth and row["predicted"] == predicted]
        mean_conf = float(np.mean([row["confidence"] for row in subset])) if subset else 0.0
        mean_margin = float(np.mean([row["margin"] for row in subset])) if subset else 0.0
        emit(
            f"  {truth:<38} -> {predicted:<38} | {count:>3} | {mean_conf * 100:8.1f}% | {mean_margin * 100:10.1f}%"
        )
    emit("")

    eager = [row for row in rows if not row["correct"] and row["confidence"] >= 0.5]
    if eager:
        emit(f"  confident mistakes (>= 50% while wrong): {len(eager)}")
        for row in sorted(eager, key=lambda item: -item["confidence"])[:top]:
            emit(
                f"    {row['image']:<34} truth={row['ground_truth']:<34} -> {row['predicted']:<34} "
                f"{row['confidence'] * 100:5.1f}%"
            )
        emit("")


def report_crop_rollup(rows: list[dict], class_names: list[str], emit) -> None:
    ev = load_evaluator()
    family_of_truth = {name: family_of(ev, name) for name in class_names}

    per_crop: dict[str, dict[str, int]] = {}
    for row in rows:
        truth_family = family_of_truth.get(row["ground_truth"], "unknown")
        bucket = per_crop.setdefault(truth_family, {"n": 0, "disease": 0, "crop": 0})
        bucket["n"] += 1
        bucket["disease"] += int(row["correct"])
        bucket["crop"] += int(family_of_truth.get(row["predicted"], "unknown") == truth_family)

    total = len(rows)
    disease_hits = sum(bucket["disease"] for bucket in per_crop.values())
    crop_hits = sum(bucket["crop"] for bucket in per_crop.values())

    emit("=" * 118)
    emit("4. CROP-LEVEL ROLLUP (does a wrong disease still name the right plant?)")
    emit("=" * 118)
    emit(f"  disease-level accuracy: {disease_hits}/{total} ({disease_hits / total * 100:.2f}%)")
    emit(f"  crop-level accuracy   : {crop_hits}/{total} ({crop_hits / total * 100:.2f}%)")
    emit(f"  off-crop errors       : {total - crop_hits} (the app shows the wrong plant entirely)")
    emit("")
    emit(f"  {'crop':<14} | {'images':>6} | {'disease':>14} | {'crop':>14}")
    for crop in sorted(per_crop, key=lambda item: per_crop[item]["crop"] / per_crop[item]["n"]):
        bucket = per_crop[crop]
        emit(
            f"  {crop:<14} | {bucket['n']:>6} | {bucket['disease']:>6}/{bucket['n']:<4} "
            f"{bucket['disease'] / bucket['n'] * 100:5.1f}% | {bucket['crop']:>6}/{bucket['n']:<4} "
            f"{bucket['crop'] / bucket['n'] * 100:5.1f}%"
        )
    emit("")


def report_worst_classes(rows: list[dict], class_names: list[str], emit, top: int) -> None:
    emit("=" * 118)
    emit(f"5. WEAKEST CLASSES (bottom {top} by accuracy at this sample size)")
    emit("=" * 118)
    scored: list[tuple[float, str, list[dict]]] = []
    for name in class_names:
        subset = [row for row in rows if row["ground_truth"] == name]
        if not subset:
            continue
        hits = sum(1 for row in subset if row["correct"])
        scored.append((hits / len(subset), name, subset))
    emit(f"  {'class':<42} | {'accuracy':>14} | {'most common wrong answer':<42}")
    for rate, name, subset in sorted(scored, key=lambda item: item[0])[:top]:
        wrong = Counter(row["predicted"] for row in subset if not row["correct"])
        beware = wrong.most_common(1)[0][0] if wrong else "-"
        count = wrong.most_common(1)[0][1] if wrong else 0
        emit(f"  {name:<42} | {sum(1 for row in subset if row['correct']):>4}/{len(subset):<4} "
             f"{rate * 100:5.1f}% | {beware}{f' (x{count})' if count > 1 else ''}")
    emit("")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder holding train/ and val/")
    parser.add_argument("--val-per-class", type=int, default=8, help="images sampled per val class")
    parser.add_argument("--top", type=int, default=15, help="rows printed per section")
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="use the Ultralytics val transform instead of the app stretch resize",
    )
    parser.add_argument("--report", type=Path, default=HERE / "class_error_report.txt")
    parser.add_argument("--csv", type=Path, default=HERE / "class_error_results.csv")
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    for required in (args.model, args.class_names):
        if not required.exists():
            raise SystemExit(f"not found: {required}")

    ev = load_evaluator()
    class_names = ev.load_class_names(args.class_names)
    val_dir = args.data_dir / "val"
    train_dir = args.data_dir / "train"
    items = load_val_items(val_dir, args.val_per_class)
    if not items:
        raise SystemExit(f"no validation images under {val_dir}")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    rows = predict_rows(args.model, class_names, items, args.center_crop)
    hits = sum(1 for row in rows if row["correct"])

    emit("=" * 118)
    emit("AgroVeyra classifier error diagnosis")
    emit("=" * 118)
    emit(f"model          : {args.model}")
    emit(f"preprocessing  : {'Ultralytics resize + center-crop' if args.center_crop else 'app parity (stretch resize)'}")
    emit(f"classes        : {len(class_names)}")
    emit(f"sample         : {len(rows)} val images ({args.val_per_class}/class) from {val_dir}")
    emit(f"top-1          : {hits}/{len(rows)} ({hits / len(rows) * 100:.2f}%)")
    emit("")

    report_balance(count_images(train_dir), count_images(val_dir), class_names, emit)
    report_prior(rows, class_names, emit, args.top)
    report_confusions(rows, emit, args.top)
    report_crop_rollup(rows, class_names, emit)
    report_worst_classes(rows, class_names, emit, args.top)

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "image",
            "ground_truth",
            "predicted",
            "confidence",
            "runner_up",
            "runner_up_confidence",
            "margin",
            "correct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"report: {args.report}", flush=True)
    print(f"csv   : {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


