from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data" / "disease"
OUTPUT_ROOT = BASE_DIR / "data" / "merged"
TRAIN_ROOT = OUTPUT_ROOT / "train"
VAL_ROOT = OUTPUT_ROOT / "val"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

SEED = 42

ANDROID_CLASS_NAMES_PATH = (
    PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "class_names.json"
)
BACKEND_CLASS_NAMES_PATH = PROJECT_ROOT / "backend" / "models" / "class_names.json"


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        [p for p in folder.rglob("*") if is_image_file(p)],
        key=lambda p: str(p).lower(),
    )


def copy_image(src: Path, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = src.name.strip()  # handle trailing spaces in filenames
    dst = dst_dir / name
    if dst.exists():
        stem = dst.stem
        suffix = dst.suffix
        counter = 1
        while True:
            dst = dst_dir / f"{stem}__{counter}{suffix}"
            if not dst.exists():
                break
            counter += 1
    shutil.copy2(src, dst)


def copy_class(src_dir: Path, dst_dir: Path) -> None:
    for image in list_image_files(src_dir):
        copy_image(image, dst_dir)


def split_images(images: list[Path], seed: int = SEED) -> tuple[list[Path], list[Path]]:
    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)

    if len(shuffled) <= 1:
        return shuffled, []

    split_index = int(len(shuffled) * 0.8)
    if split_index == 0:
        split_index = 1
    if split_index >= len(shuffled):
        split_index = len(shuffled) - 1
    return shuffled[:split_index], shuffled[split_index:]


def copy_split_class(src_dir: Path, final_name: str) -> None:
    images = list_image_files(src_dir)
    train_images, val_images = split_images(images)
    for image in train_images:
        copy_image(image, TRAIN_ROOT / final_name)
    for image in val_images:
        copy_image(image, VAL_ROOT / final_name)
    if images and not train_images:
        copy_image(images[0], TRAIN_ROOT / final_name)


# ---------------------------------------------------------------------------
# Class-name normalization
# ---------------------------------------------------------------------------

def pv_crop_simplify(crop: str) -> str:
    c = crop.strip()
    if "(" in c:
        c = c.split("(")[0].rstrip("_").strip()
    if "," in c:
        c = c.split(",")[0].strip()
    return c


def pv_class_name(raw_name: str) -> str:
    name = raw_name.strip()
    if "___" in name:
        crop, disease = name.split("___", 1)
    else:
        return name

    if "healthy" in disease.lower():
        return f"{pv_crop_simplify(crop)}___healthy"
    # keep PlantVillage non-healthy classes as-is
    return name


def strip_split_suffixes(name: str) -> str:
    n = name.strip()
    lowered = n.lower()
    changed = True
    while changed:
        changed = False
        for suffix in ("_valid", "_test", "_train"):
            if lowered.endswith(suffix):
                n = n[: -len(suffix)]
                lowered = n.lower()
                changed = True
                break
    return n


def snake_lower(name: str) -> str:
    n = name.strip()
    n = n.replace("-", " ").replace("_", " ")
    n = "_".join(part for part in n.split() if part)
    return n.lower()


def rice_disease(name: str) -> str:
    n = name.strip().replace("-", " ")
    words = [w for w in n.split() if w]
    joined = "_".join(w.lower() for w in words)
    if joined:
        joined = joined[0].upper() + joined[1:]
    return joined


def is_healthy_class(name: str) -> bool:
    return "healthy" in name.lower()


# ---------------------------------------------------------------------------
# Source mergers
# ---------------------------------------------------------------------------

def merge_plantvillage() -> None:
    pv_base = (
        DATA_DIR
        / "plantvillage"
        / "New Plant Diseases Dataset(Augmented)"
        / "New Plant Diseases Dataset(Augmented)"
    )
    for split_name, target_root in (("train", TRAIN_ROOT), ("valid", VAL_ROOT)):
        split_dir = pv_base / split_name
        if not split_dir.exists():
            print(f"[WARN] PlantVillage {split_name} not found: {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir(), key=lambda p: p.name.lower()):
            if not class_dir.is_dir():
                continue
            final_name = pv_class_name(class_dir.name)
            copy_class(class_dir, target_root / final_name)
    print("[OK] PlantVillage merged")


def merge_wheat() -> None:
    wheat_base = DATA_DIR / "wheat" / "data"
    for split_name, targets in (
        ("train", [TRAIN_ROOT]),
        ("valid", [VAL_ROOT]),
        ("test", [VAL_ROOT]),
    ):
        split_dir = wheat_base / split_name
        if not split_dir.exists():
            print(f"[WARN] Wheat {split_name} not found: {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir(), key=lambda p: p.name.lower()):
            if not class_dir.is_dir():
                continue
            if is_healthy_class(class_dir.name):
                disease = "healthy"
            else:
                disease = snake_lower(strip_split_suffixes(class_dir.name))
            final_name = f"Wheat___{disease}"
            for target in targets:
                copy_class(class_dir, target / final_name)
    print("[OK] Wheat merged")


def merge_rice() -> None:
    rice_base = DATA_DIR / "rice" / "Rice_Leaf_AUG"
    if not rice_base.exists():
        print(f"[WARN] Rice not found: {rice_base}")
        return
    for class_dir in sorted(rice_base.iterdir(), key=lambda p: p.name.lower()):
        if not class_dir.is_dir():
            continue
        if is_healthy_class(class_dir.name):
            disease = "healthy"
        else:
            disease = rice_disease(class_dir.name)
        copy_split_class(class_dir, f"Rice___{disease}")
    print("[OK] Rice merged")


def merge_cotton() -> None:
    cotton_base = DATA_DIR / "cotton" / "cotton"
    if not cotton_base.exists():
        print(f"[WARN] Cotton not found: {cotton_base}")
        return
    for class_dir in sorted(cotton_base.iterdir(), key=lambda p: p.name.lower()):
        if not class_dir.is_dir():
            continue
        if is_healthy_class(class_dir.name):
            disease = "healthy"
        else:
            disease = snake_lower(class_dir.name)
        copy_split_class(class_dir, f"Cotton___{disease}")
    print("[OK] Cotton merged")


# ---------------------------------------------------------------------------
# Summary + class_names.json
# ---------------------------------------------------------------------------

def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.rglob("*") if is_image_file(p))


def summarize_and_write_class_names() -> list[str]:
    train_classes = sorted([d.name for d in TRAIN_ROOT.iterdir() if d.is_dir()])
    val_classes = sorted([d.name for d in VAL_ROOT.iterdir() if d.is_dir()])
    all_classes = sorted(set(train_classes) | set(val_classes))

    train_total = 0
    val_total = 0
    rows: list[tuple[str, int, int]] = []
    for class_name in all_classes:
        tc = count_images(TRAIN_ROOT / class_name)
        vc = count_images(VAL_ROOT / class_name)
        train_total += tc
        val_total += vc
        rows.append((class_name, tc, vc))

    print("=" * 70)
    print("Merge summary")
    print("=" * 70)
    print(f"Total classes: {len(all_classes)}")
    print(f"Total train images: {train_total}")
    print(f"Total val images: {val_total}")
    print("Per-class count table (sorted by class name):")
    for class_name, tc, vc in rows:
        print(f"  {class_name:<55} train={tc:<6} val={vc}")

    write_class_names(all_classes)
    return all_classes


def write_class_names(class_names: list[str]) -> None:
    payload = json.dumps(sorted(class_names), indent=2) + "\n"
    ANDROID_CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    BACKEND_CLASS_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    ANDROID_CLASS_NAMES_PATH.write_text(payload, encoding="utf-8")
    BACKEND_CLASS_NAMES_PATH.write_text(payload, encoding="utf-8")
    print("[OK] Wrote class_names.json (JSON array format):")
    print(f"     {ANDROID_CLASS_NAMES_PATH}")
    print(f"     {BACKEND_CLASS_NAMES_PATH}")
    print(f"     Number of classes in new file: {len(sorted(class_names))}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if OUTPUT_ROOT.exists():
        shutil.rmtree(OUTPUT_ROOT)
    TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
    VAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("Merging PlantVillage ...")
    merge_plantvillage()
    print("Merging Wheat ...")
    merge_wheat()
    print("Merging Rice ...")
    merge_rice()
    print("Merging Cotton ...")
    merge_cotton()

    summarize_and_write_class_names()


if __name__ == "__main__":
    main()