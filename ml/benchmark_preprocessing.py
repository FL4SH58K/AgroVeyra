#!/usr/bin/env python
"""Preprocessing / TTA / confidence benchmarks for the deployed AgroVeyra TFLite model.

The evaluator (`ml/evaluate_real_world.py`) answers "is the shipped model right?" with a
single preprocessing setting. This script answers the follow-up question: *which*
preprocessing, and does test-time augmentation or a confidence gate buy anything?

It runs the deployed flatbuffer itself, so every number here transfers directly to
`TFLiteHelper.kt` and `backend/utils/predict.py`:

    ml/.venv/Scripts/python.exe ml/benchmark_preprocessing.py
    ml/.venv/Scripts/python.exe ml/benchmark_preprocessing.py --model ml/export_build_i320/agroveyra_model_float32.tflite

Transforms compared
  app                  Bitmap.createScaledBitmap -> stretch SxS (what the app does today)
  short-side-crop      Ultralytics classify_transforms (resize short side to S + centre crop S),
                       i.e. the transform the checkpoint was validated with
  short-side-1.14-crop resize short side to round(1.14*S) + centre crop S (ImageNet recipe)
  letterbox-114/-0     aspect-preserving pad to SxS, grey (Ultralytics LetterBox) or black

TTA
  none / hflip         hflip averages the softmax of the image and its mirror
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_REAL_DIR = HERE / "real_world_test"
DEFAULT_VAL_DIR = HERE / "data" / "merged" / "val"
DEFAULT_MODEL = PROJECT_ROOT / "backend" / "models" / "agroveyra_model.tflite"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "backend" / "models" / "class_names.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TRANSFORMS = ("app", "short-side-crop", "short-side-1.14-crop", "letterbox-114", "letterbox-0")
THRESHOLDS = (0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99)

# real-world file-name labels -> exact training class names
GROUND_TRUTH_ALIASES = {
    "healthy_rice": "Rice___healthy",
    "healthy_wheat": "Wheat___healthy",
    "healthy_tomato": "Tomato___healthy",
    "healthy_potato": "Potato___healthy",
    "healthy_apple": "Apple___healthy",
    "healthy_corn": "Corn___healthy",
    "healthy_maize": "Corn___healthy",
    "rice_brown_spot": "Rice___Brown_spot",
    "rice_bacterial_leaf_blight": "Rice___Bacterial_leaf_blight",
    "wheat_yellow_rust": "Wheat___yellow_rust",
    "wheat_powdery_mildew": "Wheat___mildew",
    "tomato_early_blight": "Tomato___Early_blight",
    "cotton_bacterial_blight": "Cotton___bacterial_blight",
}


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def import_interpreter():
    for module_name in ("ai_edge_litert.interpreter", "tensorflow.lite.python.interpreter"):
        try:
            return importlib.import_module(module_name).Interpreter
        except Exception:
            continue
    raise SystemExit("No TFLite interpreter available (ai_edge_litert or tensorflow).")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_class_names(path: Path) -> list[str]:
    names = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def describe_tensor(detail: dict) -> str:
    shape = [int(dim) for dim in detail["shape"]]
    scale, zero_point = detail["quantization"]
    return f"shape={shape} dtype={np.dtype(detail['dtype']).name} scale={scale:.6g} zp={zero_point}"


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return image.crop((left, top, left + size, top + size))


def _short_side(image: Image.Image, size: int) -> Image.Image:
    """torchvision T.Resize(int): scale the SHORTEST edge to `size`, keep the aspect ratio."""
    width, height = image.size
    if width <= height:
        new_width, new_height = size, max(int(round(size * height / width)), size)
    else:
        new_height, new_width = size, max(int(round(size * width / height)), size)
    return image.resize((new_width, new_height), Image.BILINEAR)


def _letterbox(image: Image.Image, size: int, fill: int) -> Image.Image:
    """Aspect-preserving pad to a size x size canvas (Ultralytics LetterBox, no stride rounding)."""
    width, height = image.size
    scale = min(size / width, size / height)
    new_width = max(int(round(width * scale)), 1)
    new_height = max(int(round(height * scale)), 1)
    resized = image.resize((new_width, new_height), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size), (fill, fill, fill))
    canvas.paste(resized, ((size - new_width) // 2, (size - new_height) // 2))
    return canvas


def apply_transform(image: Image.Image, size: int, name: str) -> np.ndarray:
    """Return a float32 NHWC [1, size, size, 3] batch scaled to [0, 1] (TFLiteHelper parity)."""
    if name == "app":
        out = image.resize((size, size), Image.BILINEAR)
    elif name == "short-side-crop":
        out = _center_crop(_short_side(image, size), size)
    elif name == "short-side-1.14-crop":
        out = _center_crop(_short_side(image, int(round(size * 1.14))), size)
    elif name == "letterbox-114":
        out = _letterbox(image, size, 114)
    elif name == "letterbox-0":
        out = _letterbox(image, size, 0)
    else:
        raise ValueError(f"unknown transform {name!r}")
    array = np.asarray(out, dtype=np.float32) / 255.0
    return array[None, ...]


class TFLiteClassifier:
    """Wrapper around the deployed flatbuffer: NHWC float32 input, 60-way softmax output."""

    def __init__(self, model_path: Path, threads: int = 4) -> None:
        interpreter_class = import_interpreter()
        self.interpreter = interpreter_class(model_path=str(model_path), num_threads=threads)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.size = int(self.input_detail["shape"][1])
        self.layout = "NHWC" if int(self.input_detail["shape"][-1]) in (1, 3) else "NCHW"
        self._in_scale, self._in_zero = self.input_detail["quantization"]
        self._out_scale, self._out_zero = self.output_detail["quantization"]

    def _feed(self, batch: np.ndarray) -> np.ndarray:
        tensor = np.transpose(batch, (0, 3, 1, 2)) if self.layout == "NCHW" else batch
        if self._in_scale:
            tensor = np.clip(np.round(tensor / self._in_scale) + self._in_zero, -128, 255)
            tensor = tensor.astype(self.input_detail["dtype"])
        else:
            tensor = tensor.astype(np.float32)
        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        raw = self.interpreter.get_tensor(self.output_detail["index"])
        scores = raw.astype(np.float32).reshape(-1)
        if self._out_scale:
            scores = (scores - self._out_zero) * self._out_scale
        scores = np.clip(scores, 0.0, None)
        total = float(scores.sum())
        if total <= 0.0:
            raise SystemExit("model emitted non-positive scores; cannot normalise")
        return scores / total

    def predict(self, batch: np.ndarray, hflip: bool = False) -> np.ndarray:
        probs = self._feed(batch)
        if hflip:
            probs = (probs + self._feed(np.ascontiguousarray(batch[:, :, ::-1, :]))) / 2.0
        return probs


def load_real_world(directory: Path) -> list[tuple[Path, str]]:
    """(path, training class name) parsed from `<crop>_<disease>_<number>.jpg`."""
    items: list[tuple[Path, str]] = []
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stem = path.stem
        head, _, tail = stem.rpartition("_")
        label = head if head and tail.isdigit() else stem
        items.append((path, GROUND_TRUTH_ALIASES.get(label.lower(), label)))
    return items


def load_val_subset(directory: Path, per_class: int) -> list[tuple[Path, str]]:
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


def evaluate(
    classifier: TFLiteClassifier,
    items: list[tuple[Path, str]],
    class_names: list[str],
    transform: str,
    hflip: bool,
) -> dict:
    index_of = {name: index for index, name in enumerate(class_names)}
    correct = 0
    confidences: list[float] = []
    gt_probabilities: list[float] = []
    rows: list[dict] = []
    for path, ground_truth in items:
        with Image.open(path) as handle:
            image = handle.convert("RGB")
        batch = apply_transform(image, classifier.size, transform)
        probs = classifier.predict(batch, hflip=hflip)
        best = int(np.argmax(probs))
        gt_index = index_of.get(ground_truth)
        hit = gt_index is not None and gt_index == best
        correct += int(hit)
        confidences.append(float(probs[best]))
        gt_probabilities.append(float(probs[gt_index]) if gt_index is not None else float("nan"))
        rows.append(
            {
                "image": path.name,
                "ground_truth": ground_truth,
                "predicted": class_names[best],
                "probability": float(probs[best]),
                "ground_truth_probability": gt_probabilities[-1],
                "correct": int(hit),
            }
        )
    total = len(items)
    return {
        "transform": transform,
        "tta": "hflip" if hflip else "none",
        "images": total,
        "correct": correct,
        "accuracy": correct / total if total else float("nan"),
        "mean_confidence": float(np.mean(confidences)) if confidences else float("nan"),
        "mean_gt_probability": float(np.nanmean(gt_probabilities)) if gt_probabilities else float("nan"),
        "rows": rows,
    }


def coverage_curve(result: dict, thresholds: tuple[float, ...]) -> list[tuple[float, int, float]]:
    """(threshold, answered, accuracy among answered) for the confidence gate."""
    curve = []
    for threshold in thresholds:
        kept = [row for row in result["rows"] if row["probability"] >= threshold]
        accuracy = sum(row["correct"] for row in kept) / len(kept) if kept else float("nan")
        curve.append((threshold, len(kept), accuracy))
    return curve


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--val-per-class", type=int, default=8, help="images sampled per val class")
    parser.add_argument("--only", default=None, help="comma-separated subset of transforms to run")
    parser.add_argument("--report", type=Path, default=HERE / "benchmark_preprocessing.txt")
    parser.add_argument("--csv", type=Path, default=HERE / "benchmark_preprocessing.csv")
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    if not args.model.exists():
        raise SystemExit(f"model not found: {args.model}")

    class_names = load_class_names(args.class_names)
    classifier = TFLiteClassifier(args.model)
    real_items = load_real_world(args.images_dir)
    val_items = load_val_subset(args.val_dir, args.val_per_class) if args.val_dir.exists() else []
    transforms = [name.strip() for name in args.only.split(",")] if args.only else list(TRANSFORMS)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit("=" * 118)
    emit("AgroVeyra preprocessing / TTA / confidence benchmark")
    emit("=" * 118)
    emit(f"model          : {args.model}")
    emit(f"model size     : {args.model.stat().st_size:,} bytes (sha256 {sha256(args.model)[:16]})")
    emit(f"model input    : {describe_tensor(classifier.input_detail)}  layout={classifier.layout}")
    emit(f"model output   : {describe_tensor(classifier.output_detail)}")
    emit(f"resize size    : {classifier.size}x{classifier.size} (read from the graph)")
    emit(f"real-world set : {args.images_dir} ({len(real_items)} images)")
    emit(f"val subset     : {args.val_dir} ({len(val_items)} images, {args.val_per_class}/class)")
    emit("")

    if real_items:
        sizes = []
        for path, _ in real_items:
            with Image.open(path) as handle:
                sizes.append((handle.width, handle.height))
        portrait = sum(1 for width, height in sizes if height > width)
        aspect = [max(width, height) / max(min(width, height), 1) for width, height in sizes]
        emit("real-world geometry (why preprocessing matters):")
        emit(f"  original sizes      : {sizes}")
        emit(f"  portrait orientation: {portrait}/{len(sizes)}")
        emit(f"  aspect ratio        : mean {np.mean(aspect):.3f}, max {np.max(aspect):.3f}")
        emit("")

    results: dict[tuple[str, str], dict] = {}
    summary_rows: list[dict] = []

    for label, items in (("real", real_items), ("val", val_items)):
        if not items:
            continue
        emit("-" * 118)
        emit(f"{label.upper()} SET - {len(items)} images")
        emit("-" * 118)
        emit(f"{'transform':<22} | {'tta':<5} | {'top-1':>6} | {'correct':>9} | {'mean conf':>9} | {'mean p(gt)':>11}")
        emit("-" * 118)
        for transform in transforms:
            for hflip in (False, True):
                result = evaluate(classifier, items, class_names, transform, hflip)
                results[(label, f"{transform}|{result['tta']}")] = result
                emit(
                    f"{transform:<22} | {result['tta']:<5} | {result['accuracy'] * 100:5.1f}% | "
                    f"{result['correct']:>4}/{result['images']:<4} | {result['mean_confidence'] * 100:8.2f}% | "
                    f"{result['mean_gt_probability'] * 100:10.2f}%"
                )
                summary_rows.append(
                    {
                        "dataset": label,
                        "transform": transform,
                        "tta": result["tta"],
                        "images": result["images"],
                        "correct": result["correct"],
                        "accuracy": f"{result['accuracy']:.6f}",
                        "mean_confidence": f"{result['mean_confidence']:.6f}",
                        "mean_gt_probability": f"{result['mean_gt_probability']:.6f}",
                    }
                )
        emit("")

    return _report(args, lines, results, summary_rows, emit)


def _report(args, lines: list[str], results: dict, summary_rows: list[dict], emit) -> int:
    baseline = results.get(("val", "app|none"))

    def ranked(dataset: str) -> list[tuple[tuple[str, str], dict]]:
        """No-TTA configurations of one dataset, best accuracy first."""
        keys = [key for key in results if key[0] == dataset and key[1].endswith("|none")]
        return sorted(((key, results[key]) for key in keys), key=lambda item: -item[1]["accuracy"])

    val_ranking = ranked("val")
    real_ranking = ranked("real")
    best_key = val_ranking[0][0] if val_ranking else None

    emit("=" * 118)
    emit("VERDICT (no TTA; each dataset ranked on its own - the two can disagree)")
    emit("=" * 118)
    for dataset, ranking in (("val", val_ranking), ("real", real_ranking)):
        if not ranking:
            continue
        other_label = "real" if dataset == "val" else "val"
        emit(f"  ranked on the {'val subset' if dataset == 'val' else 'real-world photos'} ({ranking[0][1]['images']} images):")
        for key, result in ranking:
            other = results.get((other_label, key[1]))
            other_text = (
                f"{other['accuracy'] * 100:5.1f}% ({other['correct']:>3}/{other['images']:<3})"
                if other
                else "                -"
            )
            emit(
                f"    {key[1].split('|')[0]:<22} {result['accuracy'] * 100:5.1f}% "
                f"({result['correct']:>3}/{result['images']:<3}) | {other_label:<4} {other_text}"
            )
        emit("")
    if baseline is not None and best_key is not None:
        best = results[best_key]
        emit("")
        emit(
            f"  best on val : {best_key[1].split('|')[0]} -> {best['accuracy'] * 100:.2f}% "
            f"({(best['accuracy'] - baseline['accuracy']) * 100:+.2f} pp vs app parity)"
        )
        real_baseline = results.get(("real", "app|none"))
        real_best = results.get(("real", best_key[1]))
        if real_baseline and real_best:
            emit(
                f"  best on real: {real_baseline['correct']}/{real_baseline['images']} app parity -> "
                f"{real_best['correct']}/{real_best['images']} with {best_key[1].split('|')[0]} "
                "(the val winner; the 16-photo set is too small to overrule it)"
            )
        if real_ranking and real_ranking[0][0][1] != best_key[1]:
            real_winner = real_ranking[0][0][1].split("|")[0]
            emit(
                f"  note: on real photos {real_winner} ranks first "
                f"({real_ranking[0][1]['correct']}/{real_ranking[0][1]['images']}), but it does not win on val - "
                "drop-in transforms are chosen on val, not on 16 images"
            )
    emit("")

    emit("=" * 118)
    emit("TTA EFFECT (hflip averaging, same transform)")
    emit("=" * 118)
    emit(f"  {'transform':<22} | {'val none':>9} | {'val hflip':>10} | {'real none':>10} | {'real hflip':>11}")
    for transform in TRANSFORMS:
        cells = []
        for label in ("val", "real"):
            none = results.get((label, f"{transform}|none"))
            flip = results.get((label, f"{transform}|hflip"))
            cells.append(
                (
                    f"{none['accuracy'] * 100:8.1f}%" if none else "       -",
                    f"{flip['accuracy'] * 100:9.1f}%" if flip else "        -",
                )
            )
        emit(
            f"  {transform:<22} | {cells[0][0]:>9} | {cells[0][1]:>10} | "
            f"{cells[1][0]:>10} | {cells[1][1]:>11}"
        )
    emit("")

    emit("=" * 118)
    emit("CONFIDENCE GATE (should the app refuse to answer below a threshold?)")
    emit("=" * 118)
    for label in ("val", "real"):
        for transform in TRANSFORMS:
            result = results.get((label, f"{transform}|none"))
            if result is None:
                continue
            emit(f"  {label} / {transform}:")
            emit(f"    {'threshold':>9} | {'answered':>8} | {'coverage':>8} | {'accuracy':>8}")
            for threshold, answered, accuracy in coverage_curve(result, THRESHOLDS):
                coverage = answered / result["images"] if result["images"] else float("nan")
                accuracy_text = f"{accuracy * 100:7.2f}%" if answered else "      -"
                emit(f"    {threshold:>9.2f} | {answered:>8} | {coverage * 100:7.1f}% | {accuracy_text}")
        emit("")

    detail = results.get(("real", "app|none"))
    if detail is not None:
        emit("=" * 118)
        emit("REAL-WORLD PHOTO DETAIL (app parity, no TTA)")
        emit("=" * 118)
        emit(f"{'':2}{'image':<36} | {'expected':<32} | {'predicted':<32} | {'conf':>6} | {'p(gt)':>6}")
        for row in detail["rows"]:
            flag = "  " if row["correct"] else "x "
            emit(
                f"{flag}{row['image']:<36} | {row['ground_truth']:<32} | {row['predicted']:<32} | "
                f"{row['probability'] * 100:5.1f}% | {row['ground_truth_probability'] * 100:5.1f}%"
            )
        emit("")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"report: {args.report}", flush=True)
    print(f"csv   : {args.csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
