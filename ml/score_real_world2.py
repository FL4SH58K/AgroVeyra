"""Score the supplementary real-world photos in ml/real_world_test2/.

Uses the exact app-parity pipeline from evaluate_real_world.py (bilinear stretch to
224, RGB/255, the shipped TFLite asset) but writes its own results file, so it never
clobbers the original 16-photo ml/real_world_results.csv/.txt.

Ground truth comes from the file name <crop>_<disease>_<index>.jpg via
evaluate_real_world.GROUND_TRUTH_ALIASES.

Usage:
    ml\\.venv\\Scripts\\python.exe ml\\score_real_world2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import evaluate_real_world as ev  # noqa: E402

IMAGES_DIR = HERE / "real_world_test2"
MODEL = ev.MODEL_CANDIDATES[0]
CLASS_NAMES = ev.CLASS_NAMES_CANDIDATES[0]
REPORT = HERE / "real_world_test2_results.txt"


def main() -> int:
    images = (
        sorted(
            (p for p in IMAGES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in ev.IMAGE_SUFFIXES),
            key=lambda p: str(p).lower(),
        )
        if IMAGES_DIR.is_dir()
        else []
    )
    if not images:
        print(f"no images in {IMAGES_DIR}")
        return 1

    class_names = ev.load_class_names(CLASS_NAMES)
    interpreter = ev.import_interpreter()(model_path=str(MODEL), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit("=" * 110)
    emit("AgroVeyra - supplementary real-world test (real_world_test2)")
    emit("=" * 110)
    emit(f"model  : {MODEL}")
    emit(f"images : {len(images)}")

    correct = 0
    scored = 0
    for path in images:
        truth, source = ev.resolve_ground_truth(ev.label_from_filename(path), class_names)
        if truth is None:
            emit(f"SKIP (no ground-truth alias for this filename): {path.name}")
            continue
        scored += 1
        pixels = ev.preprocess_image(path, ev.APP_INPUT_SIZE, False)
        tensor = ev.build_input_tensor(pixels, input_detail, "app")
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        scores = ev.ensure_probabilities(
            ev.dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        )
        order = np.argsort(scores)[::-1]
        top1 = class_names[int(order[0])]
        top5 = [class_names[int(i)] for i in order[:5]]
        conf = float(scores[order[0]])
        ok = top1 == truth
        correct += int(ok)

        emit("")
        emit(f"  {path.name}")
        emit(f"    truth : {truth}   (alias: {source})")
        emit(f"    top-1 : {top1}   conf {conf * 100:.2f}%   -> {'CORRECT' if ok else 'WRONG'}")
        emit("    top-5 : " + " | ".join(f"{n} ({scores[order[i]] * 100:.1f}%)" for i, n in enumerate(top5)))

    emit("")
    emit("-" * 110)
    emit(f"RESULT: {correct}/{scored} correct (top-1)")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {REPORT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
