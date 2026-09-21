"""Build the approved scoped pest dataset (Step 2 of the pest-model task).

Approved scope: IP102 classes relevant to the crops the app already diagnoses, based on IP102's
own class names and the documented principal host of each named species, minus the 8 classes with
fewer than 200 images (their per-class val counts of 7-19 images are not measurable).

  41 classes / 27,551 images -> ml/data/pest/ip102_scoped/{train,val,test}/<slug>/

Properties this script guarantees:
  * COPIES, never moves - ml/data/pest/ip102/ stays intact and untouched.
  * folder names are readable slugs (rice_leaf_roller), not IP102's 0..101 indices, so the
    classes.txt 1-indexed / folder 0-indexed offset cannot bite us at inference time.
  * pest_class_names.json is written in Ultralytics' sorted-folder order, and the script prints
    that order so it can be diffed against model.names after training.
  * refuses to run if the destination already exists (no silent merging into a stale copy).
  * a manifest with per-class per-split counts is written for the record.

Deliberately NOT in scope (documented gap): cotton, tomato, pepper and potato have no named or
host-specific class in IP102 - they are only reachable through polyphagous classes, which is a
scope the project rejected as too soft to train on.

Usage:  ml\\.venv\\Scripts\\python.exe ml\\build_pest_scoped.py
"""

import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "data" / "pest" / "ip102" / "classification"
DST = ROOT / "data" / "pest" / "ip102_scoped"
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

# (slug, ip102 folder index, crop, ip102 class name)
CLASSES = [
    ("rice_leaf_roller", 0, "rice", "rice leaf roller"),
    ("rice_leaf_caterpillar", 1, "rice", "rice leaf caterpillar"),
    ("paddy_stem_maggot", 2, "rice", "paddy stem maggot"),
    ("asiatic_rice_borer", 3, "rice", "asiatic rice borer"),
    ("yellow_rice_borer", 4, "rice", "yellow rice borer"),
    ("rice_gall_midge", 5, "rice", "rice gall midge"),
    ("rice_stemfly", 6, "rice", "Rice Stemfly"),
    ("brown_plant_hopper", 7, "rice", "brown plant hopper"),
    ("white_backed_plant_hopper", 8, "rice", "white backed plant hopper"),
    ("small_brown_plant_hopper", 9, "rice", "small brown plant hopper"),
    ("rice_water_weevil", 10, "rice", "rice water weevil"),
    ("rice_leafhopper", 11, "rice", "rice leafhopper"),
    ("rice_shell_pest", 13, "rice", "rice shell pest"),
    ("corn_borer", 22, "corn", "corn borer"),
    ("peach_borer", 26, "peach", "peach borer"),
    ("english_grain_aphid", 27, "wheat", "english grain aphid"),
    ("green_bug", 28, "wheat", "green bug"),
    ("bird_cherry_oat_aphid", 29, "wheat", "bird cherry-oataphid"),
    ("wheat_blossom_midge", 30, "wheat", "wheat blossom midge"),
    ("penthaleus_major", 31, "wheat", "penthaleus major"),
    ("wheat_phloeothrips", 33, "wheat", "wheat phloeothrips"),
    ("wheat_sawfly", 34, "wheat", "wheat sawfly"),
    ("grape_phylloxera", 59, "grape", "Viteus vitifoliae"),
    ("grape_leaf_beetle", 62, "grape", "oides decempunctata"),
    ("grape_hawk_moth", 66, "grape", "Ampelophaga"),
    ("spotted_lanternfly", 67, "grape", "Lycorma delicatula"),
    ("grape_borer", 68, "grape", "Xylotrechus"),
    ("citrus_swallowtail", 73, "citrus", "Papilio xuthus"),
    ("citrus_red_mite", 74, "citrus", "Panonchus citri McGregor"),
    ("cottony_cushion_scale", 76, "citrus", "Icerya purchasi Maskell"),
    ("arrowhead_scale", 77, "citrus", "Unaspis yanonensis"),
    ("pink_wax_scale", 78, "citrus", "Ceroplastes rubens"),
    ("florida_red_scale", 79, "citrus", "Chrysomphalus aonidum"),
    ("citrus_blackfly", 82, "citrus", "Aleurocanthus spiniferus"),
    ("citrus_fruit_fly_minax", 83, "citrus", "Tetradacus c Bactrocera minax"),
    ("oriental_fruit_fly", 84, "citrus", "Dacus dorsalis(Hendel)"),
    ("citrus_leafminer", 88, "citrus", "Phyllocnistis citrella Stainton"),
    ("black_citrus_aphid", 90, "citrus", "Toxoptera aurantii"),
    ("citrus_aphid", 91, "citrus", "Aphis citricola Vander Goot"),
    ("citrus_flatid_hopper_imitata", 94, "citrus", "Lawana imitata Melichar"),

    ("citrus_flatid_hopper_marginella", 95, "citrus", "Salurnis marginella Guerr"),
]


def main() -> int:
    if not SRC.is_dir():
        print(f"missing source: {SRC}")
        return 1
    if DST.exists():
        print(f"destination already exists, refusing to merge into it: {DST}")
        return 1

    free_gb = shutil.disk_usage(ROOT).free / 1e9
    print(f"free space: {free_gb:.1f} GB", flush=True)
    if free_gb < 5:
        print("refusing to run with under 5 GB free")
        return 1

    if len({c[0] for c in CLASSES}) != len(CLASSES):
        print("duplicate slug in CLASSES")
        return 1

    started = time.time()
    manifest = []
    total = 0
    for slug, idx, crop, name in CLASSES:
        counts = {}
        for split in SPLITS:
            src = SRC / split / str(idx)
            if not src.is_dir():
                print(f"  MISSING source split: {src}")
                counts[split] = 0
                continue
            dst = DST / split / slug
            dst.mkdir(parents=True, exist_ok=True)
            n = 0
            for image in sorted(src.iterdir()):
                if image.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                shutil.copy2(image, dst / image.name)
                n += 1
            counts[split] = n
        total += sum(counts.values())
        manifest.append((slug, idx, crop, name, counts["train"], counts["val"], counts["test"]))
        print(f"  {slug:<32} {crop:<6} train={counts['train']:>4} val={counts['val']:>3} "
              f"test={counts['test']:>4}  ({name})", flush=True)

    # Ultralytics derives class indices from the sorted folder names, so the json must match that.
    ordered = sorted(c[0] for c in CLASSES)
    json_path = DST / "pest_class_names.json"
    json_path.write_text(json.dumps(ordered, indent=2) + "\n", encoding="utf-8")

    manifest_path = ROOT / "pest_scoped_manifest.csv"
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write("index,slug,crop,ip102_folder,ip102_name,train,val,test,total\n")
        for i, (slug, idx, crop, name, tr, va, te) in enumerate(manifest):
            handle.write(f"{i},{slug},{crop},{idx},{name},{tr},{va},{te},{tr + va + te}\n")

    print("")
    print(f"copied {total} images in {time.time() - started:.0f}s -> {DST}")
    print(f"wrote {json_path.name} ({len(ordered)} classes) and {manifest_path.name}")
    print("")
    print("class index order (must equal the trained model's names):")
    for i, slug in enumerate(ordered):
        print(f"  {i:>2} {slug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
