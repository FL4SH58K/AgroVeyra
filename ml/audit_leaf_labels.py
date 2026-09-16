#!/usr/bin/env python
"""Check whether a class folder actually contains the disease it is named after.

`audit_dataset_integrity.py` finds duplicates, leakage and conflicting labels; it cannot tell you that
`val/Wheat___healthy` holds yellow-rust photographs, because that folder is internally consistent. Two
independent checkpoints agreed the images were yellow rust (`class_error_report.txt`, section 3), but a
*model-based* check is circular for training data: a model trained on mislabeled images predicts that
wrong label with full confidence.

This script uses pixels instead, so it does not depend on any checkpoint:

  * a reference set whose label is trusted (`--reference`, e.g. `Wheat___yellow_rust`);
  * a control set that must not look like the reference (`--control`, e.g. `Wheat___mildew`);
  * the class under test (`--class`, e.g. `Wheat___healthy`).

Every image becomes a normalised HSV histogram. The reference and control centroids are built from one
seeded half of their files and scored on the other half, which proves the method can separate those two
classes before it is used to judge the third. If the held-out halves are not separated cleanly the
script says so and refuses to draw a conclusion, rather than inventing evidence.

Known limitation (measured, not assumed): HSV histograms only separate *colour-distinct* classes. On the
wheat classes this method failed its own self-test - `Wheat___yellow_rust` scored 85.3% against its own
centroid while `Wheat___mildew` still scored 44.7% - so it correctly refused to judge
`Wheat___healthy` (27.1%, meaningless). Choose a colour-distinct control, or treat a suppressed verdict
as "needs human review" rather than as a negative result.

Usage:
    ml/.venv/Scripts/python.exe ml/audit_leaf_labels.py --class Wheat___healthy --reference Wheat___yellow_rust --control Wheat___mildew
    ml/.venv/Scripts/python.exe ml/audit_leaf_labels.py --class Wheat___septoria --reference Wheat___leaf_blight --control Wheat___mildew
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data" / "merged"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_EDGE = 64  # colour statistics do not need full resolution
METHOD_MIN_SEPARATION = 0.75  # reference must win this share of its own held-out images


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def features(path: Path) -> np.ndarray | None:
    """Normalised HSV histogram: stable under crop/rotation, sensitive to leaf colour."""
    try:
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((SAMPLE_EDGE, SAMPLE_EDGE))
            hsv = np.asarray(rgb.convert("HSV"), dtype=np.float32)
    except Exception:
        return None
    hue = np.histogram(hsv[..., 0], bins=24, range=(0, 256))[0].astype(np.float32)
    saturation = np.histogram(hsv[..., 1], bins=8, range=(0, 256))[0].astype(np.float32)
    value = np.histogram(hsv[..., 2], bins=8, range=(0, 256))[0].astype(np.float32)
    return np.concatenate([part / max(part.sum(), 1.0) for part in (hue, saturation, value)])


def load_features(class_dir: Path, limit: int, seed: int) -> tuple[np.ndarray, list[Path]]:
    files = sorted(file for file in class_dir.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES)
    random.Random(seed).shuffle(files)
    rows: list[np.ndarray] = []
    kept: list[Path] = []
    for path in files[:limit]:
        vector = features(path)
        if vector is not None:
            rows.append(vector)
            kept.append(path)
    if not rows:
        raise SystemExit(f"no readable images under {class_dir}")
    return np.vstack(rows), kept


def centroid(rows: np.ndarray) -> np.ndarray:
    return rows.mean(axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="folder holding train/ and val/")
    parser.add_argument("--split", default="val", choices=("train", "val"), help="split that holds --class")
    parser.add_argument("--class", dest="class_name", required=True, help="class folder whose labels are in doubt")
    parser.add_argument("--reference", required=True, help="class the suspects are accused of actually being")
    parser.add_argument("--control", required=True, help="class that must not look like the reference")
    parser.add_argument("--sample", type=int, default=300, help="images loaded per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--report", type=Path, default=HERE / "leaf_label_audit.txt")
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    class_dir = args.data_dir / args.split / args.class_name
    reference_dir = args.data_dir / "train" / args.reference
    control_dir = args.data_dir / "train" / args.control
    for directory in (class_dir, reference_dir, control_dir):
        if not directory.is_dir():
            raise SystemExit(f"not found: {directory}")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    reference_rows, _ = load_features(reference_dir, args.sample, args.seed)
    control_rows, _ = load_features(control_dir, args.sample, args.seed)
    suspect_rows, suspect_files = load_features(class_dir, args.sample, args.seed)

    # Halve the trusted sets: centroids from one half, scoring on the other (no self-scoring).
    half_reference, half_control = len(reference_rows) // 2, len(control_rows) // 2
    reference_centroid = centroid(reference_rows[:half_reference])
    control_centroid = centroid(control_rows[:half_control])

    def looks_like_reference(rows: np.ndarray) -> np.ndarray:
        to_reference = np.linalg.norm(rows - reference_centroid, axis=1)
        to_control = np.linalg.norm(rows - control_centroid, axis=1)
        return to_reference < to_control

    reference_hit = float(looks_like_reference(reference_rows[half_reference:]).mean())
    control_hit = float(looks_like_reference(control_rows[half_control:]).mean())
    suspects = looks_like_reference(suspect_rows)
    usable = reference_hit >= METHOD_MIN_SEPARATION and control_hit <= (1.0 - METHOD_MIN_SEPARATION)

    emit("=" * 118)
    emit("AgroVeyra leaf-label audit (pixel evidence, independent of any checkpoint)")
    emit("=" * 118)
    emit(f"test class   : {args.split}/{args.class_name} ({len(suspect_rows)} images scored)")
    emit(f"reference    : train/{args.reference} ({len(reference_rows)} sampled)")
    emit(f"control      : train/{args.control} ({len(control_rows)} sampled)")
    emit(f"feature      : normalised HSV histogram, {SAMPLE_EDGE}x{SAMPLE_EDGE} resize")
    emit("")
    emit("method self-test on held-out halves of the trusted classes")
    emit(f"  {args.reference:<36} scored as reference : {reference_hit * 100:5.1f}%   (expected high)")
    emit(f"  {args.control:<36} scored as reference : {control_hit * 100:5.1f}%   (expected low)")
    emit(f"  method separates the trusted classes       : {'YES' if usable else 'NO - verdict suppressed'}")
    emit("")
    emit("verdict on the class under test")
    emit(f"  images closer to `{args.reference}` than to `{args.control}` : "
         f"{int(suspects.sum())}/{len(suspects)} ({suspects.mean() * 100:.1f}%)")
    emit("")
    if not usable:
        emit("  => colour statistics cannot separate these classes; this folder needs human review.")
    elif suspects.mean() >= 0.60:
        emit(f"  => most of `{args.class_name}` looks like `{args.reference}`: the label is probably wrong.")
        emit("     Correct the source labels (or drop the folder) before expecting the model to learn it.")
    elif suspects.mean() <= 0.40:
        emit(f"  => most of `{args.class_name}` does not look like `{args.reference}`; labels look plausible.")
    else:
        emit(f"  => mixed: about half of `{args.class_name}` resembles `{args.reference}`. Inspect by eye.")
    emit("")
    emit("  furthest toward the reference (inspect these first)")
    margin = np.linalg.norm(suspect_rows - reference_centroid, axis=1) - np.linalg.norm(suspect_rows - control_centroid, axis=1)
    for index in np.argsort(margin)[:10]:
        emit(f"    {suspect_files[index].relative_to(args.data_dir)}")
    emit("")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {args.report}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
