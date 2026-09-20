"""Confidence-gate threshold sweep for the shipped AgroVeyra classifier.

Pure post-hoc analysis of evaluation CSVs already on disk - no inference, no GPU, no retrain.
Every row reuses the model's own softmax top-1 already recorded by ml/evaluate_real_world.py,
so re-running this script is free and deterministic.

Inputs
  real_world_results.csv            16 real field photos, NEW model (float32 flatbuffer)
  real_world_results_centercrop.csv same 16 photos scored with Ultralytics resize+center-crop
  val_subset_results.csv            180 files from ml/data/merged/val      (LEAKY split)
  val_subset_results_cleanval.csv   180 files from ml/data/merged_clean/val (honest split)

Output
  ml/confidence_gate_analysis.txt   sweep tables + per-threshold answer lists
"""

import csv
import os

ML = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ML, "confidence_gate_analysis.txt")

THRESHOLDS = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

FILES = [
    ("field_new", "real_world_results.csv", "16 real field photos (NEW model)"),
    ("field_crop", "real_world_results_centercrop.csv", "16 real field photos (center-crop transform)"),
    ("val_clean", "val_subset_results_cleanval.csv", "180 files, ml/data/merged_clean/val (honest)"),
    ("val_leaky", "val_subset_results.csv", "180 files, ml/data/merged/val (LEAKY)"),
]


def load(name):
    with open(os.path.join(ML, name), newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["conf"] = float(row["confidence"])
        row["ok"] = row["correct"].strip().lower() == "yes"
    return rows


def stats(rows, thr):
    """Return the gate statistics for one threshold."""
    shown = [r for r in rows if r["conf"] >= thr]
    good = [r for r in shown if r["ok"]]
    return {
        "shown": len(shown),
        "good": len(good),
        "prec": len(good) / len(shown) if shown else float("nan"),
        "cov": len(shown) / len(rows) if rows else float("nan"),
        "wrong": [r for r in shown if not r["ok"]],
        "hidden": [r for r in rows if r["ok"] and r["conf"] < thr],
        "total": len(rows),
    }


def pct(value):
    return "  n/a " if value != value else f"{value * 100:5.2f}%"


def label(row):
    return f"{row['filename']} [{row['predicted_class']} {row['conf'] * 100:.2f}%]"


def main():
    data = {key: load(name) for key, name, _ in FILES}
    lines = []
    add = lines.append

    add("=" * 100)
    add("CONFIDENCE GATE SWEEP - AgroVeyra shipped model (disease_clean, float32 re-export)")
    add("=" * 100)
    add("model     : backend/models/agroveyra_model.tflite  sha256 6aec2f1afedaf4ec  (60 classes, 224x224)")
    add("source    : post-hoc re-scoring of the CSVs produced by ml/evaluate_real_world.py --val-subset 3")
    add("            no new inference was run; nothing here depends on the GPU or on training")
    add("gate today: android/.../TFLiteHelper.kt MIN_CONFIDENCE = 0.70f (tuned on the replaced model)")
    add("            backend/main.py LOW_CONFIDENCE_THRESHOLD = 0.70 (env-overridable)")
    add("gate after: 0.95 - shipped by ml/real_world_comparison.txt section 7; see section 4 for the rule")
    add("meaning   : 'shown' = the app presents a diagnosis; below the gate it must ask for a retake")
    add("")
    for key, name, desc in FILES:
        add(f"  {key:11s} n={len(data[key]):3d}   {desc}")
    add("")

    add("=" * 100)
    add("1. SWEEP - field photos (what the user actually sees)")
    add("=" * 100)
    add("   gate    photos shown   coverage   of those CORRECT   shown WRONG   PRECISION")
    add("   ------------------------------------------------------------------------------")
    for key, name, desc in FILES[:2]:
        rows = data[key]
        add(f"   --- {desc}")
        base = stats(rows, 0.0)
        add(f"    none           {base['shown']:2d}/16      {pct(base['cov'])}          {base['good']:2d}             "
            f"{len(base['wrong']):2d}        {pct(base['prec'])}")
        for thr in THRESHOLDS:
            s = stats(rows, thr)
            add(f"    {thr:.2f}           {s['shown']:2d}/16      {pct(s['cov'])}          {s['good']:2d}             "
                f"{len(s['wrong']):2d}        {pct(s['prec'])}")
        add("")

    add("=" * 100)
    add("2. SWEEP - in-domain val sample (the gate must not start rejecting good answers)")
    add("=" * 100)
    add("   gate    cleanval shown  coverage  precision   |  leakyval shown  coverage  precision")
    add("   ---------------------------------------------------------------------------------------")
    for thr in [0.0] + THRESHOLDS:
        tag = "  none" if thr == 0.0 else f"  {thr:.2f}"
        cells = []
        for key in ("val_clean", "val_leaky"):
            s = stats(data[key], thr)
            cells.append(f"{s['shown']:3d}/180    {pct(s['cov'])}    {pct(s['prec'])}")
        add(f"   {tag}   {cells[0]}   |  {cells[1]}")

    add("")
    add("=" * 100)
    add("3. WHAT EACH GATE DOES TO THE 16 FIELD ANSWERS, PHOTO BY PHOTO")
    add("=" * 100)
    field = data["field_new"]
    for thr in THRESHOLDS:
        s = stats(field, thr)
        add(f"   --- gate {thr:.2f}   shown {s['shown']}/16   correct among shown {s['good']}   precision {pct(s['prec'])}")
        for r in s["wrong"]:
            add(f"       SHOWN BUT WRONG : {label(r)}")
        for r in s["hidden"]:
            add(f"       hidden (correct): {label(r)}")
        add("")

    add("=" * 100)
    add("4. RULE-BASED PICK")
    add("=" * 100)
    add("   rule: the LOWEST gate whose field precision is 100% while still showing at least 4 of the")
    add("         16 photos, so the app does not degenerate into an unconditional 'retake' screen.")
    add("")
    pick = None
    for thr in THRESHOLDS:
        s = stats(field, thr)
        if s["prec"] == 1.0 and s["shown"] >= 4:
            pick = (thr, s)
            break
    if pick is None:
        add("   no threshold in the sweep satisfies the rule - widen the sweep or grow the field set.")
    else:
        thr, s = pick
        vc = stats(data["val_clean"], thr)
        add(f"   pick: {thr:.2f}   field {s['shown']}/16 shown, all {s['good']} correct (100.00% precision),")
        add(f"                   honest val still shows {vc['shown']}/180 ({pct(vc['cov'])} coverage, {pct(vc['prec'])} precision)")
        for r in s["hidden"]:
            add(f"         withheld correct: {label(r)}")
    add("")
    add("   caveat: 16 photos means one photo is 6.25% - this pick is tuned on a very small field set")
    add("           and must be re-checked whenever ml/real_world_test grows. It is not overfitting to")
    add("           the val set (val is 99%+ precise at every gate), it is overfitting to 16 images.")

    add("")
    add("=" * 100)
    add("5. ROBUSTNESS - stretch vs center-crop, photo by photo")
    add("=" * 100)
    add("   the app stretches the bitmap to 224x224; center-crop is the Ultralytics-style alternative.")
    add("   a gate that only holds under one transform is tuned to the resize policy, not to the model.")
    add("")
    add("   threshold    stretch: shown/wrong   center-crop: shown/wrong")
    add("   ---------------------------------------------------------------")
    crop = {r["filename"]: r for r in data["field_crop"]}
    for thr in [0.0] + THRESHOLDS:
        a, b = stats(field, thr), stats(data["field_crop"], thr)
        tag = "  none" if thr == 0.0 else f"  {thr:.2f}"
        add(f"   {tag}              {a['shown']:2d} / {len(a['wrong']):2d}                "
            f"{b['shown']:2d} / {len(b['wrong']):2d}")
    add("")
    add("   photo (field set)                  truth -> predicted                     stretch   crop")
    add("   ------------------------------------------------------------------------------------")
    for r in sorted(field, key=lambda x: -abs(x["conf"] - crop[x["filename"]]["conf"])):
        c = crop[r["filename"]]
        add(f"   {r['filename']:33s} {r['predicted_class']:28s} {r['conf'] * 100:6.2f}%{'Y' if r['ok'] else 'n'} "
            f"{c['conf'] * 100:6.2f}%{'Y' if c['ok'] else 'n'}")
    add("")
    add("   rows sorted by how much the transform moved the confidence; Y = prediction matched truth.")
    add("   photos wrong under BOTH transforms are a model+data problem, not a gate problem.")

    text = "\n".join(lines) + "\n"
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"written: {OUT}")


if __name__ == "__main__":
    main()

