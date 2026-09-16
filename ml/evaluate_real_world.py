#!/usr/bin/env python
"""Real-world evaluation for the exported AgroVeyra TFLite classifier.

Preprocessing deliberately mirrors the on-device pipeline in
android/app/src/main/java/com/agroveyra/app/utils/TFLiteHelper.kt:

  1. decode the image as RGB
  2. stretch-resize to 224x224 with bilinear filtering
     (Bitmap.createScaledBitmap(bitmap, 224, 224, true))
  3. scale each channel to [0, 1] in RGB order (no mean/std, no BGR swap)
  4. quantize to int8/uint8 with the input tensor scale/zero-point when the
     model is quantized: round(value / scale) + zero_point
  5. dequantize the output tensor with the same formula and softmax only when
     the model emits raw logits

Ground truth comes from the file name: <crop>_<disease>_<number>.jpg

Usage:
    ml/.venv/Scripts/python.exe ml/evaluate_real_world.py
    ml/.venv/Scripts/python.exe ml/evaluate_real_world.py --center-crop
    ml/.venv/Scripts/python.exe ml/evaluate_real_world.py --layout native
    ml/.venv/Scripts/python.exe ml/evaluate_real_world.py --model path/to/model.tflite

`--center-crop` switches to the Ultralytics validation transform (resize the
short side to 256 px then center-crop) purely as a parity diagnostic; the
default mode is the one the phone actually uses and is the fair test.
`--layout` controls how the pixel buffer is arranged for the model tensor:
"app" (byte-identical to TFLiteHelper), "native" (the model's declared layout)
or "both" (default: report an accuracy for each).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

APP_INPUT_SIZE = 224

DEFAULT_IMAGES_DIR = HERE / "real_world_test"
DEFAULT_VAL_DIR = HERE / "data" / "merged" / "val"
MODEL_CANDIDATES = (
    PROJECT_ROOT / "backend" / "models" / "agroveyra_model.tflite",
    PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "agroveyra_model.tflite",
    HERE / "models" / "agroveyra_model.tflite",
)
CLASS_NAMES_CANDIDATES = (
    PROJECT_ROOT / "backend" / "models" / "class_names.json",
    PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "class_names.json",
)

RESULTS_CSV = HERE / "real_world_results.csv"
RESULTS_TXT = HERE / "real_world_results.txt"
VAL_RESULTS_CSV = HERE / "val_subset_results.csv"
VAL_RESULTS_TXT = HERE / "val_subset_results.txt"

# Labels used in the real-world test file names -> exact training class names.
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

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_stdout() -> None:
    """Make sure box-drawing/percent output never crashes on a cp1252 console."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def resolve_path(explicit: str | None, candidates, description: str) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"{description} not found: {path}")
        return path
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    tried = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise SystemExit(f"{description} not found. Tried:\n{tried}")


def load_class_names(path: Path) -> list[str]:
    import json

    names = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(names, list) or not names:
        raise SystemExit(f"No class names found in {path}")
    return [str(name) for name in names]


def normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def split_class_name(class_name: str) -> tuple[str, str]:
    if "___" in class_name:
        crop, disease = class_name.split("___", 1)
    else:
        crop, disease = "", class_name
    return crop.replace("_", " ").strip(), disease.replace("_", " ").strip()


def label_from_filename(image_path: Path) -> str:
    """cotton_bacterial_blight_2.jpg -> cotton_bacterial_blight"""
    stem = re.sub(r"[_\-\s]*\d+$", "", image_path.stem)
    return stem.strip("_- ").lower()


def resolve_ground_truth(label: str, class_names: list[str]) -> tuple[str | None, str]:
    alias = GROUND_TRUTH_ALIASES.get(label)
    if alias is not None and alias in class_names:
        return alias, "alias"

    normalized = normalize(label)
    for class_name in class_names:
        if normalize(class_name) == normalized:
            return class_name, "exact"

    tokens = label.split("_")
    if len(tokens) >= 2:
        crop = normalize(tokens[0])
        disease = normalize("_".join(tokens[1:]))
        best_name: str | None = None
        best_score = 0.0
        for class_name in class_names:
            class_crop, class_disease = split_class_name(class_name)
            if normalize(class_crop) != crop:
                continue
            score = SequenceMatcher(None, disease, normalize(class_disease)).ratio()
            if score > best_score:
                best_name, best_score = class_name, score
        if best_name is not None and best_score >= 0.5:
            return best_name, f"fuzzy({best_score:.2f})"

    return None, "unmatched"


def preprocess_image(image_path: Path, size: int, center_crop: bool) -> np.ndarray:
    """Replicate TFLiteHelper.convertBitmapToInputBuffer() pixel pipeline."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        if center_crop:
            # Ultralytics validation transform: resize the short side to 256 px,
            # then center-crop to the model input size.
            resized_side = int(round(size * 256 / 224))
            width, height = image.size
            gain = resized_side / min(width, height)
            image = image.resize(
                (
                    max(resized_side, int(round(width * gain))),
                    max(resized_side, int(round(height * gain))),
                ),
                Image.BILINEAR,
            )
            left = (image.width - size) // 2
            top = (image.height - size) // 2
            image = image.crop((left, top, left + size, top + size))
        else:
            # App parity: Bitmap.createScaledBitmap(bitmap, 224, 224, true)
            image = image.resize((size, size), Image.BILINEAR)
        pixels = np.asarray(image, dtype=np.float32)

    return np.ascontiguousarray(pixels)


def model_input_layout(shape) -> str:
    """Classify the model's declared input layout."""
    dims = [int(dim) for dim in shape]
    if len(dims) == 4:
        if dims[-1] == 3:
            return "nhwc"
        if dims[1] == 3:
            return "nchw"
    return "unknown"


def build_input_tensor(pixels: np.ndarray, input_detail: dict, layout_mode: str = "app") -> np.ndarray:
    """pixels is HWC RGB float in [0, 1] - the order TFLiteHelper produces.

    layout_mode="app"    -> the exact float stream the Android app writes into the
                            input ByteBuffer (interleaved RGB, no transposition),
                            just reshaped to the model's declared input shape.
    layout_mode="native" -> data placed in the layout the model declares
                            (e.g. HWC -> CHW when the input tensor is channels-first).
    """
    normalized = pixels / 255.0  # RGB order, no mean/std, no BGR swap (as on device)
    dtype = np.dtype(input_detail["dtype"])
    shape = [int(dim) for dim in input_detail["shape"]]
    declared = model_input_layout(shape)

    if layout_mode == "native" and declared == "nhwc":
        arranged = normalized[np.newaxis, ...]
    elif layout_mode == "native" and declared == "nchw":
        arranged = np.transpose(normalized, (2, 0, 1))[np.newaxis, ...]
    else:
        # App behaviour: a flat interleaved RGB stream; TFLite reads it in tensor order.
        arranged = normalized.reshape(-1).reshape(shape)

    if dtype in (np.dtype(np.float32), np.dtype(np.float64)):
        return arranged.astype(np.float32)

    if dtype in (np.dtype(np.int8), np.dtype(np.uint8)):
        scale, zero_point = input_detail["quantization"]
        if not scale:
            raise SystemExit("Quantized input tensor reports scale=0, cannot quantize.")
        # Same formula as TFLiteHelper.quantizeBuffer()
        quantized = np.round(arranged / scale) + zero_point
        limits = np.iinfo(dtype)
        return np.clip(quantized, limits.min, limits.max).astype(dtype)

    raise SystemExit(f"Unsupported input tensor dtype: {dtype}")


def dequantize_output(raw_scores: np.ndarray, output_detail: dict) -> np.ndarray:
    dtype = np.dtype(output_detail["dtype"])
    if dtype in (np.dtype(np.int8), np.dtype(np.uint8)):
        scale, zero_point = output_detail["quantization"]
        scores = (raw_scores.astype(np.float32) - zero_point) * scale
    else:
        scores = raw_scores.astype(np.float32)
    return scores.reshape(-1)


def ensure_probabilities(scores: np.ndarray) -> np.ndarray:
    """Ultralytics classification exports already include softmax; logits do not."""
    if float(scores.min()) >= 0.0 and 0.9 <= float(scores.sum()) <= 1.1:
        return scores
    shifted = scores - float(scores.max())
    exponent = np.exp(shifted)
    return exponent / exponent.sum()


def import_interpreter():
    errors = []
    for module_name in (
        "ai_edge_litert.interpreter",
        "tflite_runtime.interpreter",
        "tensorflow.lite",
    ):
        try:
            return getattr(importlib.import_module(module_name), "Interpreter")
        except Exception as exc:  # noqa: BLE001 - report every backend we tried
            errors.append(f"  - {module_name}: {exc}")
    raise SystemExit(
        "No TFLite runtime available. Install one, e.g.:\n"
        "  ml/.venv/Scripts/python.exe -m pip install ai-edge-litert\n"
        "Backends tried:\n" + "\n".join(errors)
    )


def describe_tensor(detail: dict) -> str:
    scale, zero_point = detail["quantization"]
    return (
        f"shape={list(detail['shape'])} dtype={np.dtype(detail['dtype']).name} "
        f"quant=({scale}, {zero_point})"
    )


def evaluate_pass(
    images, class_names, interpreter, input_detail, output_detail, args, layout_mode, truth_map=None
):
    """Run inference over every image and return one result dict per image.

    Ground truth normally comes from the file name (`<crop>_<disease>_<n>.jpg`).
    `truth_map` (image path -> (class name, truth source)) overrides that, which is how
    the model's own val split is scored: there the class is the folder name.
    """
    rows: list[dict] = []
    for image_path in images:
        label = label_from_filename(image_path)
        if truth_map and image_path in truth_map:
            ground_truth, truth_source = truth_map[image_path]
        else:
            ground_truth, truth_source = resolve_ground_truth(label, class_names)

        pixels = preprocess_image(image_path, args.input_size, args.center_crop)
        tensor = build_input_tensor(pixels, input_detail, layout_mode)
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        raw_scores = interpreter.get_tensor(output_detail["index"])
        scores = ensure_probabilities(dequantize_output(raw_scores, output_detail))

        # Descending sort by -score with a stable kind keeps the lowest class index first among
        # equal scores, which is exactly how the Android app breaks ties (IntArray.maxByOrNull
        # scans left to right and keeps the first maximum). Quantized exports emit many exact
        # ties (the deployed int8 softmax is quantized to ~1/256 steps), so this matters.
        order = np.argsort(-scores, kind="stable")
        tied_top1 = int(np.count_nonzero(scores == scores.max()))
        ranked = [
            (
                class_names[index] if index < len(class_names) else f"class_{index}",
                float(scores[index]),
            )
            for index in order[: max(3, args.top_k)]
        ]
        predicted_class, confidence = ranked[0]
        rows.append(
            {
                "filename": image_path.name,
                "label": label,
                "ground_truth": ground_truth or "",
                "truth_source": truth_source,
                "predicted_class": predicted_class,
                "confidence": confidence,
                "correct": ground_truth is not None and predicted_class == ground_truth,
                "tied_top1": tied_top1,
                "candidates": ranked,
                "top2_class": ranked[1][0],
                "top2_confidence": ranked[1][1],
                "top3_class": ranked[2][0],
                "top3_confidence": ranked[2][1],
            }
        )
    return rows


def report_pass(rows, args, emit, title: str) -> None:
    """Print the results table plus the accuracy breakdown for one pass."""
    resolved = [row for row in rows if row["ground_truth"]]
    correct = [row for row in resolved if row["correct"]]
    top_k = max(3, args.top_k)
    top_k_hits = sum(
        1
        for row in resolved
        if row["ground_truth"] in [name for name, _ in row["candidates"][:top_k]]
    )

    emit(title)
    emit("-" * 132)
    for header_line in format_table_header():
        emit(header_line)
    for row in rows:
        emit(format_table_row(row))
    emit("")

    emit(f"SUMMARY - {title}")
    emit("-" * 132)
    emit(f"images evaluated      : {len(rows)}")
    emit(f"ground truth resolved : {len(resolved)}")
    emit(f"correct top-1         : {len(correct)}")
    if resolved:
        emit(f"OVERALL ACCURACY      : {len(correct) / len(resolved) * 100:.2f}%  ({len(correct)}/{len(resolved)})")
        emit(f"top-{top_k} accuracy        : {top_k_hits / len(resolved) * 100:.2f}%  ({top_k_hits}/{len(resolved)})")
    else:
        emit("OVERALL ACCURACY      : n/a (no ground truth resolved)")
    tied = [row for row in resolved if row.get("tied_top1", 1) > 1]
    if tied:
        emit(
            f"top-1 exact ties      : {len(tied)}/{len(resolved)} images share their top score "
            "with another class (int8 output is quantized to ~1/256 steps)"
        )
    unresolved = [row["filename"] for row in rows if not row["ground_truth"]]
    if unresolved:
        emit(f"unresolved labels     : {', '.join(unresolved)}")
    emit("")

    emit("PER-LABEL ACCURACY")
    emit("-" * 132)
    for truth in sorted({row["ground_truth"] for row in resolved}):
        subset = [row for row in resolved if row["ground_truth"] == truth]
        hits = sum(1 for row in subset if row["correct"])
        emit(f"{truth:<42} {hits}/{len(subset)} correct ({hits / len(subset) * 100:5.1f}%)")
    emit("")

    mistakes = [row for row in resolved if not row["correct"]]
    emit("MISCLASSIFICATIONS (top-3 candidates)")
    emit("-" * 132)
    if not mistakes:
        emit("none")
    for row in mistakes:
        candidates = " | ".join(f"{name} {score * 100:.2f}%" for name, score in row["candidates"])
        emit(f"{row['filename']:<41} truth={row['ground_truth']:<34} -> {candidates}")
    emit("")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="path to agroveyra_model.tflite")
    parser.add_argument("--class-names", help="path to class_names.json")
    parser.add_argument(
        "--images-dir",
        default=str(DEFAULT_IMAGES_DIR),
        help="folder containing <crop>_<disease>_<number>.jpg images",
    )
    parser.add_argument("--input-size", type=int, default=APP_INPUT_SIZE)
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="use the Ultralytics val transform (resize-256 + center-crop) instead "
        "of the app stretch resize; diagnostic only",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--layout",
        choices=("app", "native", "both"),
        default="both",
        help="app = feed the exact byte stream TFLiteHelper writes (default, the fair "
        "test); native = place pixels in the layout the model declares; both = run "
        "both passes and report each accuracy",
    )
    parser.add_argument(
        "--val-subset",
        type=int,
        default=0,
        help="also score N images per class from the model's own val split (PASS 3, "
        "ground truth from the folder names); 0 disables",
    )
    parser.add_argument("--val-dir", default=str(DEFAULT_VAL_DIR), help="validation split to sample from")
    return parser.parse_args()


def format_table_header() -> list[str]:
    return [
        f"{'filename':<41} | {'ground truth':<32} | {'predicted class':<32} | {'conf':>7} | correct",
        "-" * 132,
    ]


def format_table_row(row: dict) -> str:
    truth = row["ground_truth"] or f"(unresolved '{row['label']}')"
    mark = "yes" if row["correct"] else "NO"
    return (
        f"{row['filename']:<41} | {truth:<32} | {row['predicted_class']:<32} | "
        f"{row['confidence'] * 100:6.2f}% | {mark}"
    )


def write_reports(
    rows: list[dict], lines: list[str], csv_path: Path = RESULTS_CSV, txt_path: Path = RESULTS_TXT
) -> None:
    fieldnames = [
        "filename",
        "label",
        "ground_truth",
        "truth_source",
        "predicted_class",
        "confidence",
        "correct",
        "top2_class",
        "top2_confidence",
        "top3_class",
        "top3_confidence",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            record = {key: row.get(key, "") for key in fieldnames}
            record["confidence"] = f"{row['confidence']:.6f}"
            record["correct"] = "yes" if row["correct"] else "no"
            record["top2_confidence"] = f"{row['top2_confidence']:.6f}"
            record["top3_confidence"] = f"{row['top3_confidence']:.6f}"
            writer.writerow(record)

    output = list(lines)
    output.append("")
    output.append("CSV: " + str(csv_path))
    output.append("TXT: " + str(txt_path))
    txt_path.write_text("\n".join(output) + "\n", encoding="utf-8")


def collect_val_images(val_dir: Path, class_names: list[str], per_class: int):
    """Return (images, truth_map) for the model's own val split, `per_class` images per folder."""
    images: list[Path] = []
    truth_map: dict[Path, tuple[str, str]] = {}
    if not val_dir.is_dir():
        return images, truth_map
    for folder in sorted(path for path in val_dir.iterdir() if path.is_dir()):
        ground_truth, _ = resolve_ground_truth(folder.name.lower(), class_names)
        if ground_truth is None:
            ground_truth = folder.name if folder.name in class_names else None
        for image in sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:per_class]:
            images.append(image)
            truth_map[image] = (ground_truth, f"folder:{folder.name}")
    return images, truth_map


def run_val_subset(args, class_names, interpreter, input_detail, output_detail, declared_layout) -> list[dict]:
    """Score the model on its own val split and write a separate CSV/TXT report.

    In-domain accuracy is the number that proves the export pipeline (layout, quantization,
    class order) is correct: the real-world photos carry a genuine domain gap, so they
    cannot distinguish "broken export" from "hard images".
    """
    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    images, truth_map = collect_val_images(Path(args.val_dir), class_names, args.val_subset)
    emit("=" * 132)
    emit(f"PASS 3 - MODEL'S OWN VAL SPLIT ({args.val_subset} images per class, ground truth = folder name)")
    emit("=" * 132)
    if not images:
        emit(f"no images found under {args.val_dir} - skipped")
        return []
    layout_mode = "native" if declared_layout == "nchw" else "app"
    emit(f"val dir     : {args.val_dir} ({len(images)} images)")
    emit(f"feed layout : {layout_mode} (model declares {declared_layout.upper()})")
    emit("")
    rows = evaluate_pass(
        images, class_names, interpreter, input_detail, output_detail, args, layout_mode, truth_map
    )
    report_pass(rows, args, emit, "PASS 3 - VAL SPLIT SUMMARY")
    write_reports(rows, lines, VAL_RESULTS_CSV, VAL_RESULTS_TXT)
    emit(f"saved CSV -> {VAL_RESULTS_CSV}")
    emit(f"saved TXT -> {VAL_RESULTS_TXT}")
    emit("")
    return rows


def main() -> int:
    args = parse_args()
    configure_stdout()

    model_path = resolve_path(args.model, MODEL_CANDIDATES, "TFLite model")
    class_names_path = resolve_path(args.class_names, CLASS_NAMES_CANDIDATES, "class names file")
    images_dir = Path(args.images_dir)
    if not images_dir.is_dir():
        raise SystemExit(f"Images folder not found: {images_dir}")

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise SystemExit(f"No images found in {images_dir}")

    class_names = load_class_names(class_names_path)
    interpreter_class = import_interpreter()
    interpreter = interpreter_class(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    mode = (
        "Ultralytics val transform (resize short side 256 + center crop) [diagnostic]"
        if args.center_crop
        else "app parity: bilinear stretch 224x224, RGB/255, tensor quantization"
    )
    checksum = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    emit("=" * 132)
    emit("AgroVeyra real-world evaluation - agroveyra_model.tflite")
    emit("=" * 132)
    emit(f"model         : {model_path}")
    emit(f"model size    : {model_path.stat().st_size:,} bytes (sha256 {checksum})")
    emit(f"class names   : {class_names_path} ({len(class_names)} classes)")
    emit(f"images dir    : {images_dir} ({len(images)} images)")
    emit(f"preprocessing : {mode}")
    emit(f"input tensor  : {describe_tensor(input_detail)}")
    emit(f"output tensor : {describe_tensor(output_detail)}")
    emit("")

    output_classes = int(np.prod(output_detail["shape"]))
    if output_classes != len(class_names):
        emit(
            f"WARNING: model emits {output_classes} scores but class_names.json has "
            f"{len(class_names)} entries - labels may be shifted."
        )
        emit("")

    declared_layout = model_input_layout(input_detail["shape"])
    app_rows = evaluate_pass(
        images, class_names, interpreter, input_detail, output_detail, args, "app"
    )
    report_pass(
        app_rows,
        args,
        emit,
        "PASS 1 - APP PARITY (pixels fed exactly the way TFLiteHelper feeds them)",
    )

    native_rows = None
    if args.layout in ("native", "both") and declared_layout != "nhwc":
        native_rows = evaluate_pass(
            images, class_names, interpreter, input_detail, output_detail, args, "native"
        )
        report_pass(
            native_rows,
            args,
            emit,
            f"PASS 2 - MODEL-DECLARED LAYOUT ({declared_layout.upper()}, diagnostic only)",
        )
        emit(
            "NOTE: the deployed model declares a channels-first input tensor "
            f"{[int(dim) for dim in input_detail['shape']]}, while TFLiteHelper writes "
            "NHWC-interleaved RGB floats into it."
        )
        emit(
            "      PASS 1 reproduces exactly what the phone does today; PASS 2 shows what the same "
            "model scores when the pixels are"
        )
        emit("      placed in the layout the model was exported with.")
        emit("")

    if args.val_subset > 0:
        run_val_subset(args, class_names, interpreter, input_detail, output_detail, declared_layout)

    primary_rows = native_rows if (args.layout == "native" and native_rows) else app_rows
    primary_name = "model-declared layout" if primary_rows is native_rows else "app parity"
    emit(f"CSV/TXT report written for: {primary_name}")
    write_reports(primary_rows, lines)
    emit(f"saved CSV -> {RESULTS_CSV}")
    emit(f"saved TXT -> {RESULTS_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
