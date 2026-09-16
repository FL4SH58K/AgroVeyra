#!/usr/bin/env python
"""Checkpoint-vs-flatbuffer parity over the model's own validation split.

`export_tflite_float32.py` compares the exported graph with the PyTorch checkpoint on the
8 real-world photos it needs for a smoke test. Those 8 photos cover ~6 crops, so a silent
regression in one of the other 54 classes (a dropped op, a wrong transpose, a class-order
slip) would ship unnoticed. This script answers the deployment question instead:

    "does the file we actually ship (backend/models + android assets) still score what the
     checkpoint scored, on hundreds of images across all 60 classes?"

Both engines are fed byte-identical preprocessed pixels (the app pipeline in `TFLiteHelper`),
so any disagreement is an export defect, not a preprocessing difference.

Usage:
    ml/.venv/Scripts/python.exe ml/verify_export_parity.py
    ml/.venv/Scripts/python.exe ml/verify_export_parity.py --val-per-class 20
    ml/.venv/Scripts/python.exe ml/verify_export_parity.py --model ml/export_build_i320/agroveyra_model_float32.tflite
    ml/.venv/Scripts/python.exe ml/verify_export_parity.py --center-crop     # Ultralytics val transform

Exit code is 0 when the flatbuffer is faithful, 1 when it is not, so CI can gate on it.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_CHECKPOINT = HERE / "runs" / "classify" / "runs" / "disease_classifier-3" / "weights" / "best.pt"
DEFAULT_MODEL = PROJECT_ROOT / "backend" / "models" / "agroveyra_model.tflite"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "backend" / "models" / "class_names.json"
DEFAULT_VAL_DIR = HERE / "data" / "merged" / "val"
DEFAULT_REAL_DIR = HERE / "real_world_test"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# A faithful export reproduces the checkpoint's argmax and probabilities. The gates below are
# loose enough for float re-association (measured <= 1e-5 so far) and tight enough to catch a
# real defect (a lost transpose or a quantized output moves probabilities by > 0.05).
MAX_PROBABILITY_DELTA = 1e-3
MIN_TOP1_AGREEMENT = 1.0


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_evaluator():
    """Reuse the preprocessing / dequantization code the evaluator and exporter share."""
    sys.path.insert(0, str(HERE))
    import evaluate_real_world as ev

    return ev


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


def load_real_items(directory: Path, ev, class_names: list[str]) -> list[tuple[Path, str]]:
    """(path, resolved class name) pairs for the field photos, using the evaluator's resolver."""
    items: list[tuple[Path, str]] = []
    for path in sorted(entry for entry in directory.rglob("*") if entry.suffix.lower() in IMAGE_SUFFIXES):
        resolved, _ = ev.resolve_ground_truth(ev.label_from_filename(path), class_names)
        if resolved:
            items.append((path, resolved))
    return items



class DualRunner:
    """Same pixels into both graphs: the shipped flatbuffer and the training checkpoint."""

    def __init__(self, checkpoint: Path, model_path: Path, center_crop: bool, layout: str) -> None:
        ev = load_evaluator()
        self.ev = ev
        import torch
        from ultralytics import YOLO

        self.torch = torch
        self.network = YOLO(str(checkpoint)).model.eval().cpu()

        interpreter_class = ev.import_interpreter()
        self.interpreter = interpreter_class(model_path=str(model_path), num_threads=4)
        self.interpreter.allocate_tensors()
        self.input_detail = self.interpreter.get_input_details()[0]
        self.output_detail = self.interpreter.get_output_details()[0]
        self.size = int(self.input_detail["shape"][1])
        declared = "app" if [int(dim) for dim in self.input_detail["shape"]][-1] == 3 else "native"
        self.layout = declared if layout == "auto" else layout
        self.center_crop = center_crop
        self.layout_declared = declared

    def predict(self, path: Path) -> dict:
        pixels = self.ev.preprocess_image(path, self.size, self.center_crop)
        tensor = self.ev.build_input_tensor(pixels, self.input_detail, self.layout)

        self.interpreter.set_tensor(self.input_detail["index"], tensor)
        self.interpreter.invoke()
        tflite_scores = self.ev.ensure_probabilities(
            self.ev.dequantize_output(self.interpreter.get_tensor(self.output_detail["index"]), self.output_detail)
        )

        torch_input = self.torch.from_numpy(tensor.astype(np.float32))
        if self.layout == "app":
            torch_input = torch_input.permute(0, 3, 1, 2)
        with self.torch.no_grad():
            output = self.network(torch_input)
            while isinstance(output, (tuple, list)):
                output = output[0]
            reference = output.reshape(-1).cpu().numpy().astype(np.float32)
        torch_scores = self.ev.ensure_probabilities(reference)

        return {
            "tflite_top": int(np.argmax(tflite_scores)),
            "tflite_conf": float(tflite_scores.max()),
            "torch_top": int(np.argmax(torch_scores)),
            "torch_conf": float(torch_scores.max()),
            "max_prob_delta": float(np.max(np.abs(tflite_scores - torch_scores))),
            "tflite_scores": tflite_scores,
            "torch_scores": torch_scores,
        }


def evaluate_split(runner: DualRunner, items: list[tuple[Path, str]], class_names: list[str]) -> list[dict]:
    """One row per image, with both engines' argmax, confidence and ground-truth outcome."""
    rows: list[dict] = []
    for path, ground_truth in items:
        result = runner.predict(path)
        truth_index = class_names.index(ground_truth)
        tflite_top = class_names[result["tflite_top"]]
        torch_top = class_names[result["torch_top"]]
        rows.append(
            {
                "image": path.name,
                "ground_truth": ground_truth,
                "tflite_top": tflite_top,
                "tflite_conf": result["tflite_conf"],
                "torch_top": torch_top,
                "torch_conf": result["torch_conf"],
                "max_prob_delta": result["max_prob_delta"],
                "agreement": result["tflite_top"] == result["torch_top"],
                "tflite_correct": tflite_top == ground_truth,
                "torch_correct": torch_top == ground_truth,
                "tflite_gt_prob": float(result["tflite_scores"][truth_index]),
                "torch_gt_prob": float(result["torch_scores"][truth_index]),
            }
        )
    return rows


def accuracy(rows: list[dict], field: str) -> float:
    return sum(1 for row in rows if row[field]) / len(rows) if rows else float("nan")


def print_split(rows: list[dict], class_names: list[str], title: str, emit, show_confusions: bool) -> None:
    total = len(rows)
    agreements = sum(1 for row in rows if row["agreement"])
    worst = max((row["max_prob_delta"] for row in rows), default=0.0)

    emit("-" * 118)
    emit(f"{title} - {total} images")
    emit("-" * 118)
    emit(f"  top-1 agreement            : {agreements}/{total}" + (" (100%)" if total and agreements == total else ""))
    emit(f"  max |p_tflite - p_torch|   : {worst:.6f}")
    emit(f"  accuracy, flatbuffer       : {accuracy(rows, 'tflite_correct') * 100:5.2f}%")
    emit(f"  accuracy, checkpoint       : {accuracy(rows, 'torch_correct') * 100:5.2f}%")
    emit("")

    if not show_confusions:
        return

    emit("  per-class accuracy (deployed pipeline):")
    emit(f"    {'class':<34} | {'flatbuffer':>15} | {'checkpoint':>15} | {'gap':>4}")
    for name in class_names:
        subset = [row for row in rows if row["ground_truth"] == name]
        if not subset:
            continue
        tflite_hits = sum(1 for row in subset if row["tflite_correct"])
        torch_hits = sum(1 for row in subset if row["torch_correct"])
        emit(
            f"    {name:<34} | {tflite_hits:>4}/{len(subset):<4} {tflite_hits / len(subset) * 100:5.1f}% | "
            f"{torch_hits:>4}/{len(subset):<4} {torch_hits / len(subset) * 100:5.1f}% | {abs(tflite_hits - torch_hits):>4}"
        )
    emit("")

    mistakes = sorted((row for row in rows if not row["tflite_correct"]), key=lambda item: -item["tflite_conf"])
    emit(f"  misclassifications (deployed pipeline), worst confidence first: {len(mistakes)}")
    for row in mistakes:
        emit(
            f"    truth={row['ground_truth']:<34} -> {row['tflite_top']:<34} "
            f"conf {row['tflite_conf'] * 100:5.1f}% | p(true) {row['tflite_gt_prob'] * 100:5.1f}%"
        )
    emit("")

    disagreements = [row for row in rows if not row["agreement"]]
    if disagreements:
        emit(f"  engine disagreements (flatbuffer argmax != checkpoint argmax): {len(disagreements)}")
        for row in disagreements:
            emit(
                f"    {row['image']:<34} flatbuffer={row['tflite_top']:<32} "
                f"checkpoint={row['torch_top']:<32} max dp {row['max_prob_delta']:.6f}"
            )
        emit("")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="the flatbuffer actually shipped")
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--val-dir", type=Path, default=DEFAULT_VAL_DIR)
    parser.add_argument("--val-per-class", type=int, default=8, help="images sampled per val class")
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_REAL_DIR, help="field photos (optional)")
    parser.add_argument(
        "--layout",
        choices=("auto", "app", "native"),
        default="auto",
        help="auto = the layout TFLiteHelper writes (interleaved NHWC)",
    )
    parser.add_argument(
        "--center-crop",
        action="store_true",
        help="use the Ultralytics val transform instead of the app stretch resize",
    )
    parser.add_argument("--report", type=Path, default=HERE / "export_parity_report.txt")
    parser.add_argument("--csv", type=Path, default=HERE / "export_parity_results.csv")
    return parser.parse_args()





def main() -> int:
    configure_stdout()
    args = parse_args()
    ev = load_evaluator()

    for required in (args.checkpoint, args.model, args.class_names):
        if not required.exists():
            raise SystemExit(f"not found: {required}")

    class_names = ev.load_class_names(args.class_names)
    runner = DualRunner(args.checkpoint, args.model, args.center_crop, args.layout)
    val_items = load_val_items(args.val_dir, args.val_per_class) if args.val_dir.exists() else []
    real_items = load_real_items(args.images_dir, ev, class_names) if args.images_dir.exists() else []

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit("=" * 118)
    emit("AgroVeyra export parity: shipped flatbuffer vs training checkpoint")
    emit("=" * 118)
    emit(f"checkpoint     : {args.checkpoint}")
    emit(f"                 {args.checkpoint.stat().st_size:,} bytes (sha256 {sha256(args.checkpoint)[:16]})")
    emit(f"flatbuffer     : {args.model}")
    emit(f"                 {args.model.stat().st_size:,} bytes (sha256 {sha256(args.model)[:16]})")
    emit(f"input tensor   : shape={[int(dim) for dim in runner.input_detail['shape']]} dtype={np.dtype(runner.input_detail['dtype']).name}")
    emit(f"preprocessing  : {'Ultralytics resize + center-crop' if args.center_crop else 'app parity (stretch resize)'}")
    emit(f"pixel layout   : {runner.layout} (declared by the graph: {runner.layout_declared})")
    emit(f"classes        : {len(class_names)}")
    emit(f"val subset     : {args.val_dir} ({len(val_items)} images)")
    emit(f"field photos   : {args.images_dir} ({len(real_items)} images)")
    emit("")

    val_rows = evaluate_split(runner, val_items, class_names)
    real_rows = evaluate_split(runner, real_items, class_names) if real_items else []

    if val_rows:
        print_split(val_rows, class_names, "VALIDATION SPLIT (every class sampled)", emit, show_confusions=True)
    if real_rows:
        print_split(real_rows, class_names, "FIELD PHOTOS", emit, show_confusions=True)

    all_rows = val_rows + real_rows
    if not all_rows:
        raise SystemExit("no images to compare; check --val-dir and --images-dir")

    agreements = sum(1 for row in all_rows if row["agreement"])
    worst_delta = max(row["max_prob_delta"] for row in all_rows)
    agreement_rate = agreements / len(all_rows)
    faithful = agreement_rate >= MIN_TOP1_AGREEMENT and worst_delta <= MAX_PROBABILITY_DELTA

    emit("=" * 118)
    emit("VERDICT")
    emit("=" * 118)
    emit(f"  argmax agreement     : {agreements}/{len(all_rows)} ({agreement_rate * 100:.2f}%), gate {MIN_TOP1_AGREEMENT * 100:.0f}%")
    emit(f"  max probability delta: {worst_delta:.6f}, gate {MAX_PROBABILITY_DELTA}")
    emit(f"  flatbuffer faithful  : {'YES' if faithful else 'NO'}")
    if faithful:
        emit("  => the shipped file scores what the checkpoint scores, so a field-photo drop is a")
        emit("     domain gap in the data, not an export regression.")
    else:
        emit("  => the shipped file does not reproduce the checkpoint; re-export before shipping.")
    emit("")

    if val_rows:
        missed = sorted({row["ground_truth"] for row in val_rows if not row["tflite_correct"]})
        emit(f"  val classes with at least one miss at this sample size: {len(missed)}/{len(class_names)}")
        for name in missed:
            subset = [row for row in val_rows if row["ground_truth"] == name]
            hits = sum(1 for row in subset if row["tflite_correct"])
            emit(f"    {name:<38} {hits}/{len(subset)}")
        emit("")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    fields = [
        "split",
        "image",
        "ground_truth",
        "tflite_top",
        "tflite_conf",
        "torch_top",
        "torch_conf",
        "max_prob_delta",
        "agreement",
        "tflite_correct",
        "torch_correct",
        "tflite_gt_prob",
        "torch_gt_prob",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split, rows in (("val", val_rows), ("real", real_rows)):
            for row in rows:
                writer.writerow({"split": split, **{key: row[key] for key in fields if key != "split"}})

    print(f"report: {args.report}", flush=True)
    print(f"csv   : {args.csv}", flush=True)
    return 0 if faithful else 1


if __name__ == "__main__":
    raise SystemExit(main())
