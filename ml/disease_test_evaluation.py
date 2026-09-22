"""Proper held-out evaluation of the deployed disease classifier (Option 2 protocol).

WHY THIS EXISTS
  The 97.6% reported on merged_clean/val is a *validation* number. For a test-style number, val is
  split into a TEST half and a SELECTION half and both are reported. Two facts make that defensible
  here, and both are re-verified inside this script:
    * the rebuild proved zero train/val overlap (0 identical digests, 0 source photos spanning), and
    * the val curve peaked at the FINAL epoch (0.976 at epoch 50), so patience never fired and no
      epoch was ever chosen on val - best.pt is last.pt's weights. The script asserts that.

WHY THE SPLIT IS BY SOURCE PHOTO, NOT BY FILE
  The images are PlantVillage-style augmented families of one leaf:
      01a66316-...___FREC_Scab 3003.JPG / _90deg / _180deg / _270deg
  Splitting by file would put two rotations of the SAME leaf on both sides, which is not an
  independent sample. Whole families are kept together - the rule ml/rebuild_split.py used.

WHAT IT MEASURES
  * the SHIPPED TFLite asset with the app's exact preprocessing (stretch -> 224, /255), because that
    is what the phone runs - not the .pt checkpoint;
  * top-1, top-5, per-class precision/recall/F1, macro and weighted averages, a 60x60 confusion
    matrix, top mispairs, a per-crop rollup, bootstrap 95% CIs, a healthy-vs-naming decomposition,
    a per-leaf metric that aggregates a leaf's augmented variants, and a gate sweep;
  * leakage re-check: md5 of every scored file against the whole train split, plus a dHash
    near-duplicate sweep against a deterministic sample of train;
  * the 16 real field photos, for a same-harness cross-check against ml/real_world_results.csv.

OUTPUTS
  ml/disease_test_evaluation.txt      the report
  ml/_disease_test_predictions.csv    one row per scored image
  ml/_disease_test_confusion.csv      confusion matrix

USAGE
  ml\\.venv\\Scripts\\python.exe ml\\disease_test_evaluation.py
  ml\\.venv\\Scripts\\python.exe ml\\disease_test_evaluation.py --limit 300        # smoke test
  ml\\.venv\\Scripts\\python.exe ml\\disease_test_evaluation.py --skip-leak-check
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(HERE))

VAL_DIR = HERE / "data" / "merged_clean" / "val"
TRAIN_DIR = HERE / "data" / "merged_clean" / "train"
FIELD_DIR = HERE / "real_world_test"
MODEL = PROJECT / "android" / "app" / "src" / "main" / "assets" / "agroveyra_model.tflite"
CLASS_JSON = PROJECT / "backend" / "models" / "class_names.json"
CHECKPOINT_DIR = HERE / "runs" / "disease_clean" / "weights"
PRED_CSV = HERE / "_disease_test_predictions.csv"
CONF_CSV = HERE / "_disease_test_confusion.csv"
REPORT = HERE / "disease_test_evaluation.txt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMGSZ = 224
SEED = 42
THRESHOLDS = (0.05, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
NEAR_DUP_TRAIN_SAMPLE = 20000
DHASH_LIMIT = 6

CROPS = (
    "Apple", "Blueberry", "Cherry", "Corn", "Cotton", "Grape", "Orange", "Peach", "Pepper",
    "Potato", "Raspberry", "Rice", "Soybean", "Squash", "Strawberry", "Tomato", "Wheat",
)


def crop_of(class_name: str) -> str:
    lower = class_name.lower()
    for crop in CROPS:
        if lower.startswith(crop.lower()):
            return crop
    return "other"


def source_photo(stem: str) -> str:
    """Group key for an augmented family: text before the first triple underscore.

    `01a66316-...___FREC_Scab 3003_90deg` -> `01a66316-...`; anything without the delimiter keeps
    its own name so unrelated files are never merged.
    """
    return stem.split("___", 1)[0] if "___" in stem else stem


def list_images(folder: Path) -> list[Path]:
    return sorted(
        (p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda p: str(p).lower(),
    )


def build_split(val_dir: Path, seed: int) -> tuple[list[dict], list[dict], dict]:
    """Split val along source-photo boundaries, stratified per class.

    Returns (test_rows, selection_rows, per_class_group_counts). Whole augmented families move
    together, so no leaf appears on both sides.
    """
    rng = random.Random(seed)
    test_rows: list[dict] = []
    selection_rows: list[dict] = []
    counts: dict[str, tuple[int, int]] = {}

    for class_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        groups: dict[str, list[Path]] = defaultdict(list)
        for image in list_images(class_dir):
            groups[source_photo(image.stem)].append(image)
        keys = sorted(groups)
        rng.shuffle(keys)
        n_test = len(keys) // 2
        test_keys = set(keys[:n_test])
        for key in keys:
            target = test_rows if key in test_keys else selection_rows
            for image in groups[key]:
                target.append({"class": class_dir.name, "path": image, "group": key})
        counts[class_dir.name] = (n_test, len(keys) - n_test)
    return test_rows, selection_rows, counts


def score_rows(rows: list[dict], class_names: list[str], limit: int | None, label: str) -> list[dict]:
    """Score rows through the shipped TFLite asset with the app's preprocessing."""
    import evaluate_real_world as ev

    interpreter_class = ev.import_interpreter()
    interpreter = interpreter_class(model_path=str(MODEL), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    subset = rows[:limit] if limit else rows
    started = time.time()
    out: list[dict] = []
    for index, row in enumerate(subset, 1):
        pixels = ev.preprocess_image(row["path"], IMGSZ, False)
        tensor = ev.build_input_tensor(pixels, input_detail, "app")
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        scores = ev.ensure_probabilities(
            ev.dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        )
        order = np.argsort(scores)[::-1]
        top_index = int(order[0])
        top5_names = [class_names[int(i)] for i in order[:5]]
        out.append(
            {
                "class": row["class"],
                "group": row["group"],
                "file": str(row["path"]),
                "true": row["class"],
                "pred": class_names[top_index],
                "conf": float(scores[top_index]),
                "correct": class_names[top_index] == row["class"],
                "top5": " | ".join(top5_names),
                "in_top5": row["class"] in top5_names,
            }
        )
        if index % 1000 == 0:
            print(f"  [{label}] {index}/{len(subset)} ({time.time() - started:.0f}s)", flush=True)
    print(f"  [{label}] {len(out)} images in {time.time() - started:.0f}s", flush=True)
    return out


def md5_of(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dhash_of(path: Path, size: int = 8):
    """64-bit difference hash, returned with the pixel size: (bits, (w, h))."""
    from PIL import Image

    with Image.open(path) as image:
        gray = image.convert("L")
        width, height = gray.size
        small = gray.resize((size + 1, size), Image.LANCZOS)
        pixels = small.tobytes()  # mode "L": one byte per pixel, row-major (Pillow >= 11 safe)
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            if pixels[base + col] > pixels[base + col + 1]:
                bits |= 1 << (row * size + col)
    return bits, (width, height)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def parallel_map(paths: list[Path], fn, workers: int = 12) -> list:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, paths))


def leak_check(scored: list[dict], train_dir: Path, near_dup: bool, sample: int, emit) -> dict:
    """Re-verify that nothing scored also lives in train: md5 (exact) then dHash (near)."""
    scored_paths = [Path(row["file"]) for row in scored]
    train_paths = list_images(train_dir)
    emit(f"  train images                      : {len(train_paths):,}")
    emit(f"  scored images checked             : {len(scored_paths):,}")

    started = time.time()
    train_md5 = set(parallel_map(train_paths, md5_of))
    scored_md5 = parallel_map(scored_paths, md5_of)
    exact_hits = [str(p) for p, digest in zip(scored_paths, scored_md5) if digest in train_md5]
    emit(f"  md5 pass ({time.time() - started:.0f}s): exact duplicates shared with train: {len(exact_hits)}")
    for path in exact_hits[:10]:
        emit(f"      DUPLICATE IN TRAIN: {path}")

    near_hits: list[tuple[str, str, int]] = []
    if near_dup:
        rng = random.Random(SEED)
        sample_paths = train_paths if len(train_paths) <= sample else rng.sample(train_paths, sample)
        started = time.time()
        buckets: dict[tuple[int, int], list[tuple[Path, int]]] = defaultdict(list)
        for path, (bits, size) in zip(sample_paths, parallel_map(sample_paths, dhash_of)):
            buckets[size].append((path, bits))
        for path, (bits, size) in zip(scored_paths, parallel_map(scored_paths, dhash_of)):
            for other, other_bits in buckets.get(size, []):
                distance = hamming(bits, other_bits)
                if distance <= DHASH_LIMIT:
                    near_hits.append((str(path), str(other), distance))
        emit(
            f"  dHash pass ({time.time() - started:.0f}s, train sample {len(sample_paths):,}): "
            f"near-duplicate pairs (dHash<={DHASH_LIMIT}): {len(near_hits)}"
        )
        for left, right, distance in near_hits[:10]:
            emit(f"      NEAR-DUP dHash={distance}: {Path(left).name} ~ {Path(right).name}")

    return {
        "train_images": len(train_paths),
        "exact_duplicates": len(exact_hits),
        "near_duplicates": len(near_hits),
        "near_dup_sampled": near_dup,
        "near_dup_sample_size": sample,
    }


def weight_identity() -> str:
    """Check whether best.pt and last.pt hold the same weights (i.e. no epoch chosen on val)."""
    best = CHECKPOINT_DIR / "best.pt"
    last = CHECKPOINT_DIR / "last.pt"
    if not (best.is_file() and last.is_file()):
        return "checkpoints not present - skipped"
    try:
        import torch

        first = torch.load(str(best), map_location="cpu", weights_only=False)
        second = torch.load(str(last), map_location="cpu", weights_only=False)
        sa, sb = first["model"].state_dict(), second["model"].state_dict()
        if set(sa) != set(sb):
            return "state_dict keys differ between best.pt and last.pt"
        worst = max(float((sa[key].float() - sb[key].float()).abs().max()) for key in sa)
        verdict = (
            "identical weights - no epoch was selected on val"
            if worst == 0.0
            else "weights differ - the reported epoch WAS chosen on val"
        )
        return f"{len(sa)} tensors compared, max |best - last| = {worst:g} ({verdict})"
    except Exception as error:  # noqa: BLE001 - reported, never fatal
        return f"could not compare weights: {error}"


def summarize(rows: list[dict]) -> dict:
    total = len(rows)
    correct = sum(1 for row in rows if row["correct"])
    top5 = sum(1 for row in rows if row["in_top5"])
    return {
        "n": total,
        "top1_correct": correct,
        "top1": correct / total if total else float("nan"),
        "top5": top5 / total if total else float("nan"),
    }


def per_class_metrics(rows: list[dict], class_names: list[str]) -> dict:
    """Precision/recall/F1 per class plus macro and support-weighted averages."""
    support = Counter(row["true"] for row in rows)
    predicted = Counter(row["pred"] for row in rows)
    true_positive = Counter(row["true"] for row in rows if row["correct"])

    per_class = {}
    for name in class_names:
        tp = true_positive.get(name, 0)
        fp = predicted.get(name, 0) - tp
        fn = support.get(name, 0) - tp
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        if precision == precision and recall == recall and (precision + recall) > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = float("nan")
        per_class[name] = {
            "support": support.get(name, 0),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    with_support = [m for m in per_class.values() if m["support"]]
    weights = np.array([m["support"] for m in with_support], dtype=float)
    return {
        "per_class": per_class,
        "macro_precision": float(np.mean([m["precision"] for m in with_support])),
        "macro_recall": float(np.mean([m["recall"] for m in with_support])),
        "macro_f1": float(np.mean([m["f1"] for m in with_support])),
        "weighted_f1": float(np.average([m["f1"] for m in with_support], weights=weights)),
    }


def confusion_matrix(rows: list[dict], class_names: list[str]) -> list[list[int]]:
    index = {name: position for position, name in enumerate(class_names)}
    matrix = [[0] * len(class_names) for _ in class_names]
    for row in rows:
        matrix[index[row["true"]]][index[row["pred"]]] += 1
    return matrix


def top_mispairs(rows: list[dict], limit: int = 25) -> list[tuple[str, str, int]]:
    pairs = Counter((row["true"], row["pred"]) for row in rows if not row["correct"])
    return [(true, pred, count) for (true, pred), count in pairs.most_common(limit)]


def confusion_kind(true_name: str, pred_name: str) -> str:
    return "same-crop wrong disease" if crop_of(true_name) == crop_of(pred_name) else "wrong crop"


def per_crop(rows: list[dict]) -> dict:
    buckets: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        buckets[crop_of(row["true"])].append(row)
    out = {}
    for crop, subset in sorted(buckets.items()):
        correct = sum(1 for row in subset if row["correct"])
        same_crop = sum(1 for row in subset if not row["correct"] and crop_of(row["pred"]) == crop)
        out[crop] = {
            "n": len(subset),
            "correct": correct,
            "top1": correct / len(subset),
            "errors_same_crop": same_crop,
            "errors_wrong_crop": len(subset) - correct - same_crop,
        }
    return out


def healthy_vs_naming(rows: list[dict]) -> dict:
    healthy = [row for row in rows if "healthy" in row["true"].lower()]
    diseased = [row for row in rows if "healthy" not in row["true"].lower()]
    named = [row for row in diseased if row["correct"]]
    same_crop = [
        row for row in diseased
        if not row["correct"] and crop_of(row["pred"]) == crop_of(row["true"])
    ]
    return {
        "healthy_n": len(healthy),
        "healthy_correct": sum(1 for row in healthy if row["correct"]),
        "diseased_n": len(diseased),
        "diseased_correct": len(named),
        "diseased_same_crop_wrong": len(same_crop),
        "diseased_wrong_crop": len(diseased) - len(named) - len(same_crop),
    }


def per_leaf(rows: list[dict]) -> dict:
    """Aggregate a leaf's augmented variants so each leaf counts once."""
    leaves: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        leaves[(row["true"], row["group"])].append(row)
    accuracies = [
        sum(1 for variant in variants if variant["correct"]) / len(variants)
        for variants in leaves.values()
    ]
    fully = sum(1 for value in accuracies if value == 1.0)
    return {
        "leaves": len(leaves),
        "mean_variant_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
        "leaves_fully_correct": fully,
        "leaf_level_top1": fully / len(leaves) if leaves else float("nan"),
        "mean_variants_per_leaf": float(np.mean([len(v) for v in leaves.values()])) if leaves else 0.0,
    }


def bootstrap_ci(correct: list[bool], rounds: int = 1000, seed: int = SEED) -> tuple[float, float]:
    if not correct:
        return (float("nan"), float("nan"))
    values = np.array(correct, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(values, size=len(values), replace=True).mean()) for _ in range(rounds)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def gate_sweep(rows: list[dict], thresholds=THRESHOLDS) -> list[dict]:
    total = len(rows)
    out = []
    for threshold in thresholds:
        shown = [row for row in rows if row["conf"] >= threshold]
        right = sum(1 for row in shown if row["correct"])
        out.append(
            {
                "threshold": threshold,
                "shown": len(shown),
                "coverage": len(shown) / total if total else 0.0,
                "correct": right,
                "wrong": len(shown) - right,
                "precision": right / len(shown) if shown else float("nan"),
                "recall": right / total if total else float("nan"),
            }
        )
    return out


def field_rows(class_names: list[str]) -> list[dict]:
    """The 16 real field photos, with ground truth resolved exactly as evaluate_real_world.py does."""
    import evaluate_real_world as ev

    rows = []
    if not FIELD_DIR.is_dir():
        return rows
    for path in sorted(p for p in FIELD_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        truth, _source = ev.resolve_ground_truth(ev.label_from_filename(path), class_names)
        if truth:
            rows.append({"class": truth, "path": path, "group": path.stem})
    return rows


def parse_args():
    parser = argparse.ArgumentParser(description="Held-out test evaluation of the deployed disease model.")
    parser.add_argument("--limit", type=int, default=None, help="smoke test: score only N images")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--skip-leak-check", action="store_true")
    parser.add_argument("--skip-near-dup", action="store_true", help="md5 only, no dHash sweep")
    parser.add_argument("--near-dup-sample", type=int, default=NEAR_DUP_TRAIN_SAMPLE)
    parser.add_argument("--skip-selection-half", action="store_true", help="score only the test half")
    return parser.parse_args()


def write_predictions(path: Path, groups: list[tuple[str, list[dict]]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "true", "pred", "confidence", "correct", "in_top5", "source_leaf", "file"])
        for split, rows in groups:
            for row in rows:
                writer.writerow([
                    split, row["true"], row["pred"], f"{row['conf']:.6f}",
                    int(row["correct"]), int(row["in_top5"]), row["group"], row["file"],
                ])


def write_confusion(path: Path, class_names: list[str], matrix: list[list[int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred"] + class_names)
        for name, row in zip(class_names, matrix):
            writer.writerow([name] + row)


def main() -> int:
    args = parse_args()
    started = time.time()

    sys.path.insert(0, str(HERE))
    import evaluate_real_world as ev

    class_names = ev.load_class_names(CLASS_JSON)
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit("=" * 118)
    emit("DISEASE CLASSIFIER - held-out TEST evaluation (Option 2 protocol)")
    emit("=" * 118)
    emit(f"model            : {MODEL}")
    emit(f"model sha256     : {hashlib.sha256(MODEL.read_bytes()).hexdigest()}")
    emit(f"classes          : {len(class_names)} (order from {CLASS_JSON.name})")
    emit(f"preprocessing    : the app's path - stretch to {IMGSZ} + /255, via evaluate_real_world")
    emit(f"scored           : {time.strftime('%Y-%m-%d %H:%M:%S')}")
    emit("")

    test_rows, selection_rows, group_counts = build_split(VAL_DIR, args.seed)
    test_leaves = len({(r["class"], r["group"]) for r in test_rows})
    sel_leaves = len({(r["class"], r["group"]) for r in selection_rows})
    emit("SPLIT (by source-photo family, stratified per class)")
    emit(f"  val images            : {len(test_rows) + len(selection_rows):,}")
    emit(f"  TEST half             : {len(test_rows):,} images / {test_leaves:,} source leaves")
    emit(f"  SELECTION half        : {len(selection_rows):,} images / {sel_leaves:,} source leaves")
    emit(f"  rule                  : whole augmented families move together, seed {args.seed}")
    thin = [name for name, (a, b) in group_counts.items() if min(a, b) < 4]
    emit(f"  classes with <4 leaves in one half: {len(thin)}" + (f" -> {', '.join(thin[:6])}" if thin else ""))
    emit("")

    emit("MODEL-SELECTION CHECK (is the reported checkpoint 'the best epoch on val'?)")
    emit(f"  best.pt vs last.pt    : {weight_identity()}")
    emit("")

    emit("SCORING")
    test_scored = score_rows(test_rows, class_names, args.limit, "test half")
    selection_scored: list[dict] = []
    if not args.skip_selection_half and not args.limit:
        selection_scored = score_rows(selection_rows, class_names, None, "selection half")
    field_scored = [] if args.limit else score_rows(field_rows(class_names), class_names, None, "field photos")
    emit("")

    if args.limit:
        summary = summarize(test_scored)
        emit(f"--limit {args.limit}: smoke run only, no leak check and no report written.")
        emit(f"  top-1 {summary['top1_correct']}/{summary['n']} = {summary['top1'] * 100:.2f}%")
        return 0

    leak = {"skipped": True}
    if not args.skip_leak_check:
        emit("LEAKAGE RE-CHECK (scored files vs the train split)")
        leak = leak_check(
            test_scored + selection_scored, TRAIN_DIR,
            near_dup=not args.skip_near_dup, sample=args.near_dup_sample, emit=emit,
        )
        emit("")
    return finish(args, class_names, test_scored, selection_scored, field_scored, leak, lines, emit, started)


def finish(args, class_names, test_scored, selection_scored, field_scored, leak, lines, emit, started) -> int:
    test_summary = summarize(test_scored)
    selection_summary = summarize(selection_scored) if selection_scored else None
    val_summary = summarize(test_scored + selection_scored)
    field_summary = summarize(field_scored) if field_scored else None

    metrics = per_class_metrics(test_scored, class_names)
    low_ci, high_ci = bootstrap_ci([row["correct"] for row in test_scored], rounds=args.bootstrap)
    matrix = confusion_matrix(test_scored, class_names)
    mispairs = top_mispairs(test_scored)
    crops = per_crop(test_scored)
    healthy = healthy_vs_naming(test_scored)
    leaves = per_leaf(test_scored)
    gates = gate_sweep(test_scored)
    field_gates = gate_sweep(field_scored) if field_scored else []

    write_predictions(
        PRED_CSV,
        [("test-half", test_scored), ("selection-half", selection_scored), ("field", field_scored)],
    )
    write_confusion(CONF_CSV, class_names, matrix)

    emit("=" * 118)
    emit("HEADLINE - held-out TEST half (the number to quote, with the caveats below)")
    emit("=" * 118)
    emit(f"  TEST half      : {test_summary['top1_correct']:,}/{test_summary['n']:,} = "
         f"{test_summary['top1'] * 100:.2f}% top-1   {test_summary['top5'] * 100:.2f}% top-5   "
         f"(95% CI {low_ci * 100:.2f}-{high_ci * 100:.2f}%)")
    if selection_summary:
        emit(f"  selection half : {selection_summary['top1_correct']:,}/{selection_summary['n']:,} = "
             f"{selection_summary['top1'] * 100:.2f}% top-1   {selection_summary['top5'] * 100:.2f}% top-5")
    emit(f"  whole val      : {val_summary['top1_correct']:,}/{val_summary['n']:,} = "
         f"{val_summary['top1'] * 100:.2f}% top-1   {val_summary['top5'] * 100:.2f}% top-5")
    if field_summary:
        emit(f"  field photos   : {field_summary['top1_correct']}/{field_summary['n']} = "
             f"{field_summary['top1'] * 100:.2f}% top-1   {field_summary['top5'] * 100:.2f}% top-5")
    emit("")
    emit(f"  macro precision {metrics['macro_precision'] * 100:.2f}%   macro recall "
         f"{metrics['macro_recall'] * 100:.2f}%   macro F1 {metrics['macro_f1'] * 100:.2f}%   "
         f"weighted F1 {metrics['weighted_f1'] * 100:.2f}%")
    emit(f"  per-leaf view  : {leaves['leaves']:,} source leaves, mean variant accuracy "
         f"{leaves['mean_variant_accuracy'] * 100:.2f}%, leaves fully correct "
         f"{leaves['leaf_level_top1'] * 100:.2f}% ({leaves['mean_variants_per_leaf']:.1f} variants/leaf)")
    emit("")
    emit("HEALTHY vs NAMING (test half)")
    emit(f"  healthy images : {healthy['healthy_correct']:,}/{healthy['healthy_n']:,} = "
         f"{100.0 * healthy['healthy_correct'] / max(healthy['healthy_n'], 1):.2f}%")
    emit(f"  diseased images: {healthy['diseased_correct']:,}/{healthy['diseased_n']:,} = "
         f"{100.0 * healthy['diseased_correct'] / max(healthy['diseased_n'], 1):.2f}%")
    emit(f"  diseased errors: {healthy['diseased_same_crop_wrong']:,} same-crop wrong disease, "
         f"{healthy['diseased_wrong_crop']:,} wrong crop")
    emit("")
    emit("PER-CROP (test half)")
    emit(f"  {'crop':<12} {'n':>7} {'top-1':>8} {'same-crop errors':>18} {'wrong-crop errors':>19}")
    for crop, data in crops.items():
        emit(f"  {crop:<12} {data['n']:>7,} {data['top1'] * 100:>7.2f}% {data['errors_same_crop']:>18,} "
             f"{data['errors_wrong_crop']:>19,}")
    emit("")
    emit("TOP MISPREDICTIONS (test half)")
    for true_name, pred_name, count in mispairs[:15]:
        emit(f"  {count:>5} x  {true_name}  ->  {pred_name}   [{confusion_kind(true_name, pred_name)}]")
    emit("")
    emit("GATE SWEEP (test half): coverage vs precision at each threshold")
    emit(f"  {'gate':>6} {'shown':>9} {'coverage':>9} {'correct':>8} {'wrong':>7} {'precision':>10} {'recall':>8}")
    for row in gates:
        emit(f"  {row['threshold']:>6.2f} {row['shown']:>9,} {row['coverage'] * 100:>8.2f}% "
             f"{row['correct']:>8,} {row['wrong']:>7,} {row['precision'] * 100:>9.2f}% "
             f"{row['recall'] * 100:>7.2f}%")
    emit("")
    if field_gates:
        emit("GATE SWEEP (the 16 real field photos, for comparison)")
        for row in field_gates:
            precision = "n/a" if not row["shown"] else f"{row['precision'] * 100:.1f}%"
            emit(f"  gate {row['threshold']:.2f}: shown {row['shown']:>2}/16, precision {precision:>6}")
    emit("")
    emit("WORST CLASSES by recall (at least 20 test images)")
    worst = sorted(
        ((name, data) for name, data in metrics["per_class"].items() if data["support"] >= 20),
        key=lambda item: item[1]["recall"],
    )[:12]
    for name, data in worst:
        emit(f"  {data['recall'] * 100:>6.2f}% recall (n={data['support']:>5,})  P {data['precision'] * 100:>6.2f}%  "
             f"F1 {data['f1'] * 100:>6.2f}%  {name}")
    emit("")
    emit("LEAKAGE RE-CHECK SUMMARY")
    if leak.get("skipped"):
        emit("  SKIPPED for this run (--skip-leak-check)")
    else:
        emit(f"  scored images checked against {leak['train_images']:,} train images")
        emit(f"  exact duplicates shared with train : {leak['exact_duplicates']}")
        if leak["near_dup_sampled"]:
            emit(f"  near-duplicate pairs (dHash<={DHASH_LIMIT}, train sample {leak['near_dup_sample_size']:,}): "
                 f"{leak['near_duplicates']}")
        else:
            emit("  near-duplicate sweep              : skipped")
    emit("")
    emit("=" * 118)
    emit("CAVEATS - read before quoting the number")
    emit("=" * 118)
    emit("  1. The TEST half shares no source leaf with the SELECTION half, but both come from the val")
    emit("     split that was monitored during training for early stopping. The model-selection check")
    emit("     above reports whether best.pt is merely the last epoch: if the weights are identical, no")
    emit("     epoch was chosen on val and this is a genuine held-out number.")
    emit("  2. Val images are PlantVillage-style augmented families - several rotations of one leaf on a")
    emit("     plain background. In-domain accuracy therefore overstates a phone photo of a leaf in a")
    emit("     field; the field-photo row above and ml/field_failure_analysis.txt are the honest pair.")
    emit("  3. Everything is scored through the SHIPPED TFLite asset with the app's preprocessing, so it")
    emit("     is the number the phone produces, not the checkpoint's best-case score.")
    emit("")
    emit(f"predictions : {PRED_CSV}")
    emit(f"confusion   : {CONF_CSV}")
    emit(f"elapsed     : {time.time() - started:.0f}s")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.name} ({len(lines)} lines)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
