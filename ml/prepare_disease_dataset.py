from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

SEED = 42
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BASE_DIR / "dataset" / "disease"
TRAIN_ROOT = OUTPUT_ROOT / "train"
VAL_ROOT = OUTPUT_ROOT / "val"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(folder: Path):
    return sorted(
        [path for path in folder.rglob("*") if is_image_file(path)],
        key=lambda item: str(item).lower(),
    )


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return len(iter_image_files(folder))


def erase_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def unique_destination_path(dest_dir: Path, file_name: str, source_file: Path) -> Path:
    candidate = dest_dir / file_name
    if not candidate.exists():
        return candidate

    stem = Path(file_name).stem
    suffix = Path(file_name).suffix
    source_tag = source_file.stem.replace(" ", "_")
    candidate = dest_dir / f"{stem}__src_{source_tag}{suffix}"
    if not candidate.exists():
        return candidate

    counter = 1
    while True:
        candidate = dest_dir / f"{stem}__src_{source_tag}__{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def copy_class_contents(source_dir: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for image in iter_image_files(source_dir):
        target_path = unique_destination_path(dest_dir, image.name, image)
        shutil.copy2(image, target_path)


def split_class_images(class_name: str, images: list[Path], train_root: Path, val_root: Path, seed: int = SEED) -> tuple[int, int]:
    rng = random.Random(seed)
    shuffled = list(images)
    rng.shuffle(shuffled)

    if len(shuffled) == 1:
        train_files = shuffled
        val_files = []
    else:
        split_index = int(len(shuffled) * 0.8)
        if split_index == 0:
            split_index = 1
        if split_index >= len(shuffled):
            split_index = len(shuffled) - 1
        train_files = shuffled[:split_index]
        val_files = shuffled[split_index:]

    train_dir = train_root / class_name
    val_dir = val_root / class_name
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for file_path in train_files:
        target_path = unique_destination_path(train_dir, file_path.name, file_path)
        shutil.copy2(file_path, target_path)
    for file_path in val_files:
        target_path = unique_destination_path(val_dir, file_path.name, file_path)
        shutil.copy2(file_path, target_path)

    return len(train_files), len(val_files)


def normalize_wheat_name(raw_name: str) -> str:
    normalized = raw_name.strip().lower()
    for suffix in ("_train", "_valid", "_test"):
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")
    normalized = "_".join(part for part in normalized.split("_") if part)
    return normalized


def canonical_wheat_name(raw_name: str) -> str:
    cleaned = raw_name.strip()
    for suffix in ("_train", "_valid", "_test"):
        if cleaned.lower().endswith(suffix):
            cleaned = cleaned[: -len(suffix)]
    cleaned = cleaned.replace("_", " ")
    cleaned = " ".join(cleaned.split())
    return f"Wheat_{cleaned.replace(' ', '_')}"


def build_plantvillage(debug: bool = False) -> None:
    source_root = BASE_DIR / "data" / "disease" / "plantvillage" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
    debug_classes = {
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Tomato___Spider_mites Two-spotted_spider_mite",
    }

    for split_name, target_root in (("train", TRAIN_ROOT), ("valid", VAL_ROOT)):
        split_dir = source_root / split_name
        if not split_dir.exists():
            print(f"[WARN] PlantVillage {split_name} split not found: {split_dir}")
            continue
        for class_dir in sorted(split_dir.iterdir(), key=lambda item: item.name.lower()):
            if not class_dir.is_dir():
                continue

            if debug and class_dir.name in debug_classes:
                print(f"[DEBUG] PlantVillage class check: split={split_name}, class={class_dir.name}")
                print(f"[DEBUG]   source path: {class_dir}")
                print(f"[DEBUG]   exists: {class_dir.exists()}")

            copy_class_contents(class_dir, target_root / class_dir.name)


def build_rice() -> None:
    source_root = BASE_DIR / "data" / "disease" / "rice" / "Rice_Leaf_AUG"
    class_map = {
        "Bacterial Leaf Blight": "Rice_Bacterial_Leaf_Blight",
        "Brown Spot": "Rice_Brown_Spot",
        "Healthy Rice Leaf": "Rice_Healthy_Rice_Leaf",
        "Leaf Blast": "Rice_Leaf_Blast",
        "Leaf scald": "Rice_Leaf_scald",
        "Sheath Blight": "Rice_Sheath_Blight",
    }

    for raw_name, final_name in class_map.items():
        source_dir = source_root / raw_name
        if not source_dir.exists():
            print(f"[WARN] Rice class missing: {source_dir}")
            continue
        images = iter_image_files(source_dir)
        split_class_images(final_name, images, TRAIN_ROOT, VAL_ROOT, seed=SEED)


def build_wheat() -> None:
    source_root = BASE_DIR / "data" / "disease" / "wheat" / "data"
    split_dirs = [source_root / "train", source_root / "valid", source_root / "test"]

    normalized_to_source_folders: dict[str, list[Path]] = defaultdict(list)
    normalized_to_images: dict[str, list[Path]] = defaultdict(list)
    normalized_to_canonical: dict[str, str] = {}

    for split_dir in split_dirs:
        if not split_dir.exists():
            continue
        for class_dir in sorted(split_dir.iterdir(), key=lambda item: item.name.lower()):
            if not class_dir.is_dir():
                continue
            normalized_key = normalize_wheat_name(class_dir.name)
            normalized_to_source_folders[normalized_key].append(class_dir)
            normalized_to_images[normalized_key].extend(iter_image_files(class_dir))
            if normalized_key not in normalized_to_canonical:
                normalized_to_canonical[normalized_key] = canonical_wheat_name(class_dir.name)

    # Prefer the train-folder casing for the final canonical class name when available.
    train_dir = source_root / "train"
    if train_dir.exists():
        for class_dir in sorted(train_dir.iterdir(), key=lambda item: item.name.lower()):
            if not class_dir.is_dir():
                continue
            normalized_key = normalize_wheat_name(class_dir.name)
            normalized_to_canonical[normalized_key] = canonical_wheat_name(class_dir.name)

    final_mapping: dict[str, list[str]] = {}
    for normalized_key, folders in sorted(normalized_to_source_folders.items()):
        final_name = normalized_to_canonical.get(normalized_key, f"Wheat_{normalized_key}")
        final_mapping[final_name] = [str(folder.relative_to(source_root)) for folder in folders]

    print("\nWheat merge mapping table")
    print("=" * 120)
    print(f"{'Final class':<30} {'Source folders'}")
    print("-" * 120)
    for final_name in sorted(final_mapping):
        source_folders = " | ".join(final_mapping[final_name])
        print(f"{final_name:<30} {source_folders}")

    for normalized_key, images in sorted(normalized_to_images.items()):
        final_name = normalized_to_canonical.get(normalized_key, f"Wheat_{normalized_key}")
        split_class_images(final_name, images, TRAIN_ROOT, VAL_ROOT, seed=SEED)


def build_cotton(debug: bool = False) -> None:
    source_root = BASE_DIR / "data" / "disease" / "cotton" / "cotton"
    class_map = {
        "bacterial_blight": "Cotton_bacterial_blight",
        "curl_virus": "Cotton_curl_virus",
        "fussarium_wilt": "Cotton_fussarium_wilt",
        "healthy": "Cotton_healthy",
    }

    for raw_name, final_name in class_map.items():
        source_dir = source_root / raw_name
        if not source_dir.exists():
            print(f"[WARN] Cotton class missing: {source_dir}")
            continue
        images = iter_image_files(source_dir)
        if debug:
            print(f"[DEBUG] Cotton source files: class={raw_name}, found={len(images)}, source={source_dir}")
        split_class_images(final_name, images, TRAIN_ROOT, VAL_ROOT, seed=SEED)


def summarize_dataset() -> None:
    train_classes = sorted([path.name for path in TRAIN_ROOT.iterdir() if path.is_dir()])
    val_classes = sorted([path.name for path in VAL_ROOT.iterdir() if path.is_dir()])
    unique_classes = sorted(set(train_classes) | set(val_classes))

    train_total = sum(count_images(TRAIN_ROOT / class_name) for class_name in unique_classes)
    val_total = sum(count_images(VAL_ROOT / class_name) for class_name in unique_classes)

    print("\nFinal summary")
    print("=" * 80)
    print(f"Total classes: {len(unique_classes)}")
    print(f"len(final_class_list): {len(unique_classes)}")
    if len(unique_classes) != len(set(unique_classes)):
        duplicates: list[str] = []
        seen: set[str] = set()
        for class_name in unique_classes:
            if class_name in seen:
                duplicates.append(class_name)
            seen.add(class_name)
        print(f"[WARN] Duplicate class names found: {sorted(set(duplicates))}")
    else:
        print("Class list consistency check: PASS")
    print(f"Total train images: {train_total}")
    print(f"Total val images: {val_total}")
    print("Alphabetical class list:")
    for class_name in unique_classes:
        train_count = count_images(TRAIN_ROOT / class_name)
        val_count = count_images(VAL_ROOT / class_name)
        total_count = train_count + val_count
        status = "LOW DATA" if total_count < 100 else ""
        if status:
            print(f"  - {class_name} ({total_count} total)  [{status}]")
        else:
            print(f"  - {class_name} ({total_count} total)")

    low_data = []
    for class_name in unique_classes:
        total_count = count_images(TRAIN_ROOT / class_name) + count_images(VAL_ROOT / class_name)
        if total_count < 100:
            low_data.append((class_name, total_count))

    if low_data:
        print("\nLOW DATA classes:")
        for class_name, count in sorted(low_data):
            print(f"  - {class_name}: {count} total images")
    else:
        print("\nLOW DATA classes: none")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare unified disease train/val dataset.")
    parser.add_argument("--debug", action="store_true", help="Enable detailed diagnostic output for dataset path and count checks.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    erase_output_dir(TRAIN_ROOT)
    erase_output_dir(VAL_ROOT)

    build_plantvillage(debug=args.debug)
    build_rice()
    build_wheat()
    build_cotton(debug=args.debug)

    summarize_dataset()


if __name__ == "__main__":
    main()
