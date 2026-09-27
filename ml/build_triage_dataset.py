"""Build the AgroVeyra 3-way triage dataset (healthy / disease / pest_damage).

Third additive capability alongside the disease (60-class) and pest (41-class)
models. Triage answers: healthy / pathogen damage / pest damage, then routes to the
disease model, the pest-category questionnaire, or "healthy".

Sources
    healthy     -> merged_clean `*___healthy` (15 crops), uniform subsample
    disease     -> merged_clean non-healthy, EXCLUDING Wheat___aphid/mite/stem_fly
                   (arthropod classes already in the disease model) so disease ==
                   pathogen and pest_damage == arthropod.
    pest_damage -> LeLePhid aphid-presence (335) + sesame "Insect Leaf Damage" (724).

Pipeline: collect -> uniform subsample (healthy/disease) -> md5 dedup -> dHash
near-dup dedup -> 80/20 stratified split (seed 42) -> data/triage/{train,val} ->
triage_class_names.json (sorted folder order == softmax index order).

Usage:
    ml/.venv/Scripts/python.exe ml/build_triage_dataset.py             # report only
    ml/.venv/Scripts/python.exe ml/build_triage_dataset.py --apply     # write data/triage
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGED_CLEAN = HERE / "data" / "merged_clean"
DOWNLOADS = Path("C:/Users/smroh/Downloads")

PEST_CLASSES_IN_DISEASE = {"Wheat___aphid", "Wheat___mite", "Wheat___stem_fly"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42
TARGET_DEFAULT = 800
CLASS_NAMES = ["healthy", "disease", "pest_damage"]
OUTPUT_ROOT = HERE / "data" / "triage"
TRAIN_ROOT = OUTPUT_ROOT / "train"
VAL_ROOT = OUTPUT_ROOT / "val"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted([p for p in folder.rglob("*") if is_image_file(p)], key=lambda p: str(p).lower())


def find_lelephid_images() -> Path:
    for root in DOWNLOADS.iterdir():
        if root.is_dir() and root.name.lower().startswith("lelephid"):
            for d in root.rglob("Images"):
                if d.is_dir():
                    return d
    raise FileNotFoundError("LeLePhid Images folder not found under Downloads")


def find_sesame_images() -> Path:
    sesame = DOWNLOADS / "_sesame_tmp" / "insect_damage"
    if not sesame.is_dir():
        raise FileNotFoundError(f"sesame insect_damage folder not found: {sesame}")
    return sesame


def source_tag_for(path: Path) -> str:
    s = str(path).lower()
    if "lelephid" in s:
        return "llp"
    if "insect_damage" in s:
        return "ses"
    return "pv"


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def dhash_of(path: Path) -> int:
    from PIL import Image
    img = Image.open(path).convert("L").resize((9, 8), Image.LANCZOS)
    px = img.tobytes()
    h = 0
    for row in range(8):
        for col in range(8):
            h = (h << 1) | (1 if px[row * 9 + col] > px[row * 9 + col + 1] else 0)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def collect_healthy() -> dict[str, list[Path]]:
    by_crop: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        split_dir = MERGED_CLEAN / split
        if not split_dir.is_dir():
            continue
        for cls in sorted(split_dir.iterdir()):
            if cls.is_dir() and cls.name.endswith("___healthy"):
                by_crop.setdefault(cls.name, []).extend(list_images(cls))
    return by_crop


def collect_disease() -> dict[str, list[Path]]:
    by_crop: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        split_dir = MERGED_CLEAN / split
        if not split_dir.is_dir():
            continue
        for cls in sorted(split_dir.iterdir()):
            if not cls.is_dir():
                continue
            if cls.name.endswith("___healthy") or cls.name in PEST_CLASSES_IN_DISEASE:
                continue
            by_crop.setdefault(cls.name, []).extend(list_images(cls))
    return by_crop


def collect_pest_damage() -> list[Path]:
    images: list[Path] = []
    llp = find_lelephid_images()
    for p in llp.glob("*"):
        if is_image_file(p) and p.name.lower().startswith("aphid."):
            images.append(p)
    images.extend(list_images(find_sesame_images()))
    return images


def subsample_uniform(by_class: dict[str, list[Path]], target: int, seed: int) -> list[Path]:
    classes = sorted(by_class)
    n = len(classes)
    if n == 0:
        return []
    per = target // n
    remainder = target - per * n
    rng = random.Random(seed)
    selected: list[Path] = []
    for i, cls in enumerate(classes):
        imgs = by_class[cls]
        rng.shuffle(imgs)
        take = min(per + (1 if i < remainder else 0), len(imgs))
        selected.extend(imgs[:take])
    return selected


def dedup_md5(pool: list[tuple[Path, str]]) -> list[tuple[Path, str]]:
    seen: dict[str, Path] = {}
    out: list[tuple[Path, str]] = []
    for path, label in pool:
        digest = md5_of(path)
        if digest in seen:
            continue
        seen[digest] = path
        out.append((path, label))
    return out


def dedup_dhash(pool: list[tuple[Path, str]], threshold: int) -> list[tuple[Path, str]]:
    hashed = [(path, label, dhash_of(path)) for path, label in pool]
    kept: list[tuple[Path, str, int]] = []
    for item in hashed:
        if any(hamming(item[2], k[2]) <= threshold for k in kept):
            continue
        kept.append(item)
    return [(p, l) for p, l, _ in kept]


def write_class_names() -> None:
    (OUTPUT_ROOT / "triage_class_names.json").write_text(
        json.dumps(sorted(CLASS_NAMES), indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the AgroVeyra triage dataset.")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--target", type=int, default=TARGET_DEFAULT)
    ap.add_argument("--dhash-threshold", type=int, default=5)
    ap.add_argument("--no-dhash", action="store_true")
    args = ap.parse_args()

    print("=" * 96)
    print("Triage dataset plan")
    print("=" * 96)

    healthy_by = collect_healthy()
    disease_by = collect_disease()
    pest = collect_pest_damage()

    print(f"  healthy crops : {len(healthy_by)}")
    print(f"  disease crops : {len(disease_by)}")
    print(f"  pest images   : {len(pest)}")

    healthy = subsample_uniform(healthy_by, args.target, SEED)
    disease = subsample_uniform(disease_by, args.target, SEED)
    print(f"  healthy after subsample : {len(healthy)}")
    print(f"  disease after subsample : {len(disease)}")

    pool = (
        [(p, "healthy") for p in healthy]
        + [(p, "disease") for p in disease]
        + [(p, "pest_damage") for p in pest]
    )
    before = len(pool)
    pool = dedup_md5(pool)
    after_md5 = len(pool)
    if not args.no_dhash:
        pool = dedup_dhash(pool, args.dhash_threshold)
    after_dhash = len(pool)

    print(f"  pool before dedup      : {before}")
    print(f"  after md5              : {after_md5} (removed {before - after_md5})")
    print(f"  after dHash            : {after_dhash} (removed {after_md5 - after_dhash})")

    by_label: dict[str, list[Path]] = defaultdict(list)
    for p, l in pool:
        by_label[l].append(p)

    if args.apply:
        if OUTPUT_ROOT.exists():
            shutil.rmtree(OUTPUT_ROOT)
        TRAIN_ROOT.mkdir(parents=True, exist_ok=True)
        VAL_ROOT.mkdir(parents=True, exist_ok=True)

    print("-" * 96)
    print("  split (80/20 stratified, seed 42):")
    total_train = total_val = 0
    for label in CLASS_NAMES:
        rng = random.Random(SEED)
        shuffled = list(by_label[label])
        rng.shuffle(shuffled)
        idx = int(len(shuffled) * 0.8)
        if idx == 0:
            idx = 1
        if idx >= len(shuffled):
            idx = len(shuffled) - 1
        train, val = shuffled[:idx], shuffled[idx:]
        print(f"    {label:<12} train={len(train):<5} val={len(val)}")
        total_train += len(train)
        total_val += len(val)
        if args.apply:
            for split, files in (("train", train), ("val", val)):
                dst_dir = (TRAIN_ROOT if split == "train" else VAL_ROOT) / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                for i, src in enumerate(files):
                    tag = source_tag_for(src)
                    dst = dst_dir / f"{tag}_{label}_{i:04d}{src.suffix.lower()}"
                    shutil.copy2(src, dst)

    print("-" * 96)
    print(f"  TOTAL: train={total_train} val={total_val} ({total_train + total_val} images, {len(CLASS_NAMES)} classes)")

    if args.apply:
        write_class_names()
        print(f"  wrote: {OUTPUT_ROOT}")
        print(f"  wrote: {OUTPUT_ROOT / 'triage_class_names.json'} -> {sorted(CLASS_NAMES)}")
    else:
        print("  (dry run: nothing written; re-run with --apply to build data/triage)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
