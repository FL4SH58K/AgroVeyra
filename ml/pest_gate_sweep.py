"""Confidence-gate sweep for the pest classifier, measured on the IP102 test split.

IN-DOMAIN ONLY. Unlike the disease gate - which was measured on 16 real field photos - there is no
real-world pest photo set on this machine. So this table answers one specific question well ("at what
threshold is the pest model still worth showing an answer for, on IP102-style insect close-ups") and
says nothing about farmer photos.

Method: every image in ml/data/pest/ip102_scoped/test is scored ONCE through the shipped TFLite asset
using the app's preprocessing (stretch to 224, /255, no mean/std), the scores are cached to CSV, and
the threshold sweep is then applied post-hoc. That keeps it cheap and means the gate cannot influence
what was measured.

Writes ml/pest_confidence_gate_analysis.txt and ml/_pest_test_scores.csv (intermediate, gitignored).
"""

import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

SCALED = HERE / "data" / "pest" / "ip102_scoped"
NAMES_JSON = SCALED / "pest_class_names.json"
MODEL = HERE.parent / "android" / "app" / "src" / "main" / "assets" / "agroveyra_pest_model.tflite"
SCORES_CSV = HERE / "_pest_test_scores.csv"
REPORT = HERE / "pest_confidence_gate_analysis.txt"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
IMGSZ = 224

# crop per class slug, for the per-crop view (source: ml/pest_scope.py mapping)
CROP = {
    "rice": ["rice_leaf_roller", "rice_leaf_caterpillar", "paddy_stem_maggot", "asiatic_rice_borer",
             "yellow_rice_borer", "rice_gall_midge", "rice_stemfly", "brown_plant_hopper",
             "white_backed_plant_hopper", "small_brown_plant_hopper", "rice_water_weevil",
             "rice_leafhopper", "rice_shell_pest"],
    "wheat": ["english_grain_aphid", "green_bug", "bird_cherry_oat_aphid", "wheat_blossom_midge",
              "penthaleus_major", "wheat_phloeothrips", "wheat_sawfly"],
    "grape": ["grape_phylloxera", "grape_leaf_beetle", "grape_hawk_moth", "spotted_lanternfly",
              "grape_borer"],
    "citrus": ["citrus_swallowtail", "citrus_red_mite", "cottony_cushion_scale", "arrowhead_scale",
               "pink_wax_scale", "florida_red_scale", "citrus_blackfly", "citrus_fruit_fly_minax",
               "oriental_fruit_fly", "citrus_leafminer", "black_citrus_aphid", "citrus_aphid",
               "citrus_flatid_hopper_imitata", "citrus_flatid_hopper_marginella"],
    "corn": ["corn_borer"],
    "peach": ["peach_borer"],
}
THRESHOLDS = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99]


def crop_of(slug: str) -> str:
    for crop, slugs in CROP.items():
        if slug in slugs:
            return crop
    return "?"


def score_all():
    """Score every test image once; returns list of (true_slug, pred_slug, probability)."""
    if SCORES_CSV.is_file():
        rows = list(csv.DictReader(SCORES_CSV.open(encoding="utf-8")))
        if rows:
            print(f"reusing {len(rows)} cached scores from {SCORES_CSV.name}", flush=True)
            return [(r["true"], r["pred"], float(r["prob"])) for r in rows]

    import evaluate_real_world as ev

    names = json.loads(NAMES_JSON.read_text(encoding="utf-8"))
    interpreter_class = ev.import_interpreter()
    interpreter = interpreter_class(model_path=str(MODEL), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    images = []
    for class_dir in sorted((SCALED / "test").iterdir()):
        if not class_dir.is_dir():
            continue
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                images.append((class_dir.name, path))

    started = time.time()
    results = []
    for index, (truth, path) in enumerate(images, 1):
        pixels = ev.preprocess_image(path, IMGSZ, False)
        tensor = ev.build_input_tensor(pixels, input_detail, "app")
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        scores = ev.ensure_probabilities(
            ev.dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        )
        top = int(np.argmax(scores))
        results.append((truth, names[top], float(scores[top])))
        if index % 1000 == 0:
            print(f"  scored {index}/{len(images)} ({time.time() - started:.0f}s)", flush=True)

    with SCORES_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true", "pred", "prob"])
        writer.writerows(results)
    print(f"scored {len(results)} images in {time.time() - started:.0f}s", flush=True)
    return results


def main() -> int:
    results = score_all()
    total = len(results)
    correct = sum(1 for truth, pred, _ in results if truth == pred)

    out = []
    add = out.append
    add("PEST CONFIDENCE GATE - threshold vs coverage vs precision")
    add("generated by ml/pest_gate_sweep.py")
    add("")
    add("SCOPE WARNING: this is measured on the IP102 test split only. There are no real-world pest")
    add("photographs on this machine, so unlike the disease gate (16 real field photos) this table says")
    add("nothing about farmer photos. IP102 images are close-ups OF THE INSECT, and the model has no")
    add("'no pest present' class, so the gate is a UX guard here, not a correctness guarantee.")
    add("")
    add(f"model    : agroveyra_pest_model.tflite (41 classes, float32, NHWC 224)")
    add(f"test set : {total} images, all 41 classes, never used for training or checkpoint selection")
    add(f"ungated  : top-1 {correct}/{total} = {100.0 * correct / total:.2f}%")
    add("")
    add("=" * 104)
    add("THRESHOLD SWEEP (post-hoc on the same scored images; 'shown' = the app would display a name)")
    add("=" * 104)
    add(f"{'gate':>6} {'shown':>7} {'coverage':>9} {'correct':>8} {'wrong':>6} {'precision':>10} "
        f"{'recall':>8}")
    for threshold in THRESHOLDS:
        shown = [(t, p) for t, p, prob in results if prob >= threshold]
        right = sum(1 for t, p in shown if t == p)
        precision = 100.0 * right / len(shown) if shown else float("nan")
        recall = 100.0 * right / total
        add(f"{threshold:>6.2f} {len(shown):>7} {100.0 * len(shown) / total:>8.2f}% {right:>8} "
            f"{len(shown) - right:>6} {precision:>9.2f}% {recall:>7.2f}%")

    add("")
    add("=" * 104)
    add("PER-CROP VIEW at a few thresholds")
    add("=" * 104)
    for threshold in (0.50, 0.70, 0.80, 0.90, 0.95):
        add("")
        add(f"gate {threshold:.2f}:")
        for crop in sorted(CROP):
            subset = [(t, p, prob) for t, p, prob in results if crop_of(t) == crop]
            if not subset:
                continue
            shown = [(t, p) for t, p, prob in subset if prob >= threshold]
            right = sum(1 for t, p in shown if t == p)
            all_right = sum(1 for t, p, _ in subset if t == p)
            add(f"  {crop:<7} n={len(subset):>5}  ungated {100.0 * all_right / len(subset):>6.2f}%  "
                f"shown {len(shown):>5}  precision "
                f"{'n/a' if not shown else f'{100.0 * right / len(shown):.2f}%':>7}")

    add("")
    add("=" * 104)
    add("READ THIS")
    add("=" * 104)
    add("The disease model ships a 0.95 gate because at 0.95 every answer it showed was correct. The")
    add("pest model is a harder task (41 fine-grained insect species, a nano backbone), so expect the")
    add("opposite trade: a high gate buys precision and costs most of the coverage. Pick the threshold")
    add("from the table above against what the app should do when it is not confident - the same")
    add("decision the disease model faced in ml/confidence_gate_analysis.txt.")
    add("")
    add("Reminder: whatever threshold is chosen, it has NOT been validated on real farmer photos.")

    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {REPORT.name}", flush=True)
    for threshold in (0.70, 0.90, 0.95):
        shown = [(t, p) for t, p, prob in results if prob >= threshold]
        right = sum(1 for t, p in shown if t == p)
        print(f"  gate {threshold:.2f}: shown {len(shown)}/{total} "
              f"({100.0 * len(shown) / total:.1f}%), precision "
              f"{'n/a' if not shown else f'{100.0 * right / len(shown):.1f}%'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
