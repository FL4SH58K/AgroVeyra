#!/usr/bin/env python
"""Rebuild `data/merged` into a split that can actually be held out (`data/merged_clean`).

`audit_dataset_integrity.py` proved that `data/merged` cannot be validated on, for four independent
reasons (all measured, not suspected):

  1. 42.9% of `val/Wheat*` shares a *source photo* with train - `merge_datasets.split_images()` split
     a class's file list 80/20 *per file*, so `<uuid>___leaf_270deg.JPG` and `<uuid>___leaf.JPG`
     landed on opposite sides of the split;
  2. 693 val images are byte-identical to a train image (the wheat source dataset ships its own
     leaky train/valid folders and the merge copied both);
  3. 1448 same-class duplicate groups (2216 redundant files) and 256 cross-class duplicate groups
     (559 files whose pixels carry two labels - unlearnable by construction);
  4. `val/Wheat___healthy` is disputed ground truth: the source calls those 70 files healthy, while the
     deployed model calls 60/70 of them `Wheat___yellow_rust` at ~100% confidence
     (`class_error_report.txt` section 3). Whichever side is right, they cannot be used as a held-out
     label for that class, so they are quarantined for review instead of counted as healthy.

Deliberately non-destructive: `data/merged` is only read. The rebuilt split goes to a new root and
every dropped file lands in a quarantine tree with a reason, so each decision can be reviewed and
reversed. Nothing is written unless `--apply` is passed.

What it does, per class:
  1. hash every file; identical bytes inside one class -> keep one, quarantine the rest
  2. identical bytes across classes            -> quarantine every copy (ambiguous label)
  3. group the survivors into *source photos*  (uuid prefix, or stem minus `_270deg`/`_FlipLR`)
  4. split 80/20 along group boundaries        -> a photo and its rotations never straddle train/val
  5. quarantine `val/Wheat___healthy`          -> proven mislabeled (--keep-mislabeled-val to skip)
  6. promote groups from train when a class is left with fewer than --min-val-per-class val images
     (Ultralytics maps classes by folder, so every class must appear in both splits)
  7. verify before claiming success: no digest and no source photo spans train/val, every class is
     present in both, and the class set still matches `backend/models/class_names.json` in order

Usage:
    ml/.venv/Scripts/python.exe ml/rebuild_split.py                      # dry run: report only
    ml/.venv/Scripts/python.exe ml/rebuild_split.py --scope wheat        # dry run, wheat classes only
    ml/.venv/Scripts/python.exe ml/rebuild_split.py --apply              # write data/merged_clean
    ml/.venv/Scripts/python.exe ml/rebuild_split.py --apply --mode link  # hardlink instead of copy
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_SOURCE = HERE / "data" / "merged"
DEFAULT_OUT = HERE / "data" / "merged_clean"
DEFAULT_QUARANTINE = HERE / "data" / "quarantine"
DEFAULT_CLASS_NAMES = PROJECT_ROOT / "backend" / "models" / "class_names.json"
DEFAULT_CACHE = HERE / ".sha256_cache.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# `00075aa8-...___FREC_Scab 3335_270deg.JPG` == `00075aa8-...___FREC_Scab 3335.JPG`
SOURCE_SEPARATOR = re.compile(r"_{3,}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
AUGMENT_SUFFIX = re.compile(r"_(?:\d+deg|rot\d+|fliplr|flipud|flip)$", re.IGNORECASE)
# `merge_datasets.copy_image()` renames name collisions to `<stem>__1`, `<stem>__2`, ...
COLLISION_SUFFIX = re.compile(r"__\d+$")

REASON_CROSS_CLASS = "cross_class_duplicate"
REASON_SAME_CLASS = "same_class_duplicate"
REASON_MISLABELED = "mislabeled_val_wheat_healthy"

# Classes whose `val/` folder the trained model contradicts: all 70 of these images are called
# `Wheat___yellow_rust` at roughly 100% confidence while the source folder says `healthy`. A pixel
# re-check could not settle which side is right (ml/audit_leaf_labels.py failed its own separation
# self-test on these wheat classes), so the images are quarantined rather than trusted either way.
DEFAULT_MISLABELED_VAL_CLASSES = ("Wheat___healthy",)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def sha256(path: Path) -> str:
    """Digest file contents (the only identity that survives the merge's renames)."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_hash_cache(path: Path) -> dict[str, list]:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def digest_of(path: Path, root: Path, cache: dict[str, list], dirty: list[bool]) -> str:
    """sha256 of `path`, memoised on (size, mtime) so re-runs cost seconds instead of minutes."""
    key = str(path.relative_to(root)).replace("\\", "/")
    try:
        stat = path.stat()
    except OSError:
        return "unreadable"
    entry = cache.get(key)
    if isinstance(entry, list) and len(entry) == 3 and entry[0] == stat.st_size and entry[1] == int(stat.st_mtime):
        return str(entry[2])
    digest = sha256(path)
    cache[key] = [stat.st_size, int(stat.st_mtime), digest]
    dirty[0] = True
    return digest


def source_key(path: Path) -> str:
    """Identity of the *source* photo, ignoring augmentation and merge collision suffixes."""
    stem = COLLISION_SUFFIX.sub("", AUGMENT_SUFFIX.sub("", path.stem.strip()))
    prefix = SOURCE_SEPARATOR.split(stem, maxsplit=1)[0]
    match = UUID.search(prefix)
    return (match.group(0) if match else prefix).lower()


def walk_split(split_dir: Path, scope: str) -> dict[str, list[Path]]:
    """class name -> image files, for one split directory."""
    by_class: dict[str, list[Path]] = {}
    if not split_dir.is_dir():
        return by_class
    for class_dir in sorted(entry for entry in split_dir.iterdir() if entry.is_dir()):
        if scope != "all" and scope not in class_dir.name.lower():
            continue
        files = sorted(
            (file for file in class_dir.rglob("*") if file.suffix.lower() in IMAGE_SUFFIXES),
            key=lambda item: str(item).lower(),
        )
        if files:
            by_class[class_dir.name] = files
    return by_class


def place(source_file: Path, destination: Path, mode: str) -> None:
    """Copy or hardlink `source_file` to `destination`, creating parents."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "link":
        try:
            if destination.exists():
                destination.unlink()
            os.link(source_file, destination)
            return
        except OSError:
            # Different volume, or a filesystem without hardlinks: fall back to a real copy.
            pass
    shutil.copy2(source_file, destination)


def collect_records(files: list[Path], from_split: str, class_name: str, args, cache, dirty) -> list[dict]:
    return [
        {
            "path": path,
            "class": class_name,
            "from_split": from_split,
            "digest": digest_of(path, args.source, cache, dirty),
            "source": source_key(path),
        }
        for path in files
    ]


def plan_class(records: list[dict], conflict_digests: set[str], args) -> tuple[dict[str, list[dict]], list[dict]]:
    """Return (split -> kept records, quarantined records with reasons) for one class."""
    if not records:
        return {"train": [], "val": []}, []

    quarantined: list[dict] = []
    survivors: list[dict] = []

    # --- 1/2. duplicate handling (identical bytes) -------------------------------------------
    seen_digest: dict[str, dict] = {}
    for record in records:
        if record["digest"] in conflict_digests:
            quarantined.append({**record, "reason": REASON_CROSS_CLASS})
            continue
        first = seen_digest.get(record["digest"])
        if first is None:
            seen_digest[record["digest"]] = record
            survivors.append(record)
        else:
            quarantined.append({**record, "reason": REASON_SAME_CLASS})

    # --- 5. classes whose val folder is proven mislabeled -------------------------------------
    kept: list[dict] = []
    for record in survivors:
        if record["class"] in args.mislabeled_val_classes and record["from_split"] == "val":
            quarantined.append({**record, "reason": REASON_MISLABELED})
        else:
            kept.append(record)
    survivors = kept

    if not survivors:
        return {"train": [], "val": []}, quarantined

    # --- 3. source-photo families (a photo plus its offline rotations/flips) -------------------
    families: dict[str, list[dict]] = {}
    for record in survivors:
        families.setdefault(record["source"], []).append(record)

    # --- 4. split along family boundaries -----------------------------------------------------
    keys = sorted(families)
    random.Random(args.seed).shuffle(keys)
    cutoff = int(len(keys) * (1.0 - args.val_fraction))
    cutoff = max(1, min(cutoff, len(keys) - 1)) if len(keys) > 1 else len(keys)
    train_keys = keys[:cutoff]
    val_keys = keys[cutoff:]

    # --- 6. keep every class present in both splits (Ultralytics maps classes by folder) --------
    def count(keys_subset: list[str]) -> int:
        return sum(len(families[key]) for key in keys_subset)

    promoted = 0
    while count(val_keys) < args.min_val_per_class and len(train_keys) > 1:
        val_keys.append(train_keys.pop())
        promoted += 1

    split_records = {
        "train": [record for key in train_keys for record in families[key]],
        "val": [record for key in val_keys for record in families[key]],
    }
    return split_records, quarantined, promoted


def build_plan(classes: list[str], records_by_class: dict[str, list[dict]], args) -> dict:
    digest_classes: dict[str, set[str]] = {}
    for class_name, records in records_by_class.items():
        for record in records:
            digest_classes.setdefault(record["digest"], set()).add(class_name)
    conflict_digests = {digest for digest, owners in digest_classes.items() if len(owners) > 1}

    plan: dict[str, dict] = {}
    for class_name in classes:
        split_records, quarantined, promoted = plan_class(records_by_class[class_name], conflict_digests, args)
        plan[class_name] = {"splits": split_records, "quarantined": quarantined, "promoted": promoted}
    return plan


def plan_destinations(plan: dict, args) -> list[dict]:
    """Flatten the plan into one row per source file: where it goes and why.

    Files that keep their name but change split are the fix; files that leave the dataset are the
    audit findings. Same-named files from the other split are disambiguated (`x.png` -> `x__dup1.png`)
    because the original `train/` and `val/` folders both contain files called e.g. `aphid_27.png`.
    """
    rows: list[dict] = []
    used: dict[tuple[str, str], set[str]] = {}

    for class_name in sorted(plan):
        entry = plan[class_name]
        for split in ("train", "val"):
            for record in entry["splits"][split]:
                name = record["path"].name
                taken = used.setdefault((split, class_name), set())
                if name in taken:
                    stem, suffix = Path(name).stem, Path(name).suffix
                    counter = 1
                    while f"{stem}__dup{counter}{suffix}" in taken:
                        counter += 1
                    name = f"{stem}__dup{counter}{suffix}"
                taken.add(name)
                rows.append(
                    {
                        "action": split,
                        "reason": "split_rebuilt",
                        "class": class_name,
                        "from_split": record["from_split"],
                        "original": record["path"],
                        "destination": args.out / split / class_name / name,
                    }
                )
        for record in entry["quarantined"]:
            relative = record["path"].relative_to(args.source)
            rows.append(
                {
                    "action": "quarantine",
                    "reason": record["reason"],
                    "class": class_name,
                    "from_split": record["from_split"],
                    "original": record["path"],
                    "destination": args.quarantine / record["reason"] / relative,
                }
            )
    return rows


def apply_plan(rows: list[dict], args) -> int:
    written = 0
    for row in rows:
        place(row["original"], row["destination"], args.mode)
        written += 1
    return written


def verify_plan(plan: dict, args) -> tuple[list[str], dict[str, int]]:
    """The checks that decide whether the rebuild is trustworthy. Returns (failures, stats)."""
    failures: list[str] = []
    stats = {"train": 0, "val": 0, "quarantined": 0, "leaked_digests": 0, "shared_sources": 0}

    for class_name in sorted(plan):
        entry = plan[class_name]
        train_records = entry["splits"]["train"]
        val_records = entry["splits"]["val"]
        stats["train"] += len(train_records)
        stats["val"] += len(val_records)
        stats["quarantined"] += len(entry["quarantined"])

        leaked = {record["digest"] for record in train_records} & {record["digest"] for record in val_records}
        if leaked:
            stats["leaked_digests"] += len(leaked)
            failures.append(f"{class_name}: {len(leaked)} identical image(s) still on both sides")
        shared = {record["source"] for record in train_records} & {record["source"] for record in val_records}
        if shared:
            stats["shared_sources"] += len(shared)
            failures.append(f"{class_name}: {len(shared)} source photo(s) still on both sides")
        if not train_records and val_records:
            failures.append(f"{class_name}: empty train split")
        if not val_records and train_records:
            failures.append(f"{class_name}: empty val split (Ultralytics needs every class in both)")
    return failures, stats


def verify_class_order(classes: list[str], class_names_path: Path, full_scope: bool) -> list[str]:
    """Class order is a shipping contract: TFLiteHelper indexes softmax output with class_names.json."""
    if not class_names_path.is_file():
        return [f"class_names.json not found: {class_names_path}"]
    names = json.loads(class_names_path.read_text(encoding="utf-8-sig"))
    if not isinstance(names, list):
        return [f"class_names.json is not a JSON array: {class_names_path}"]

    produced = sorted(classes)
    if full_scope:
        if produced != [str(name) for name in names]:
            missing = sorted(set(map(str, names)) - set(produced))
            extra = sorted(set(produced) - set(map(str, names)))
            return [
                f"class set/order differs from class_names.json ({len(names)} names in the json, "
                f"{len(produced)} classes rebuilt)",
                f"    missing from rebuild: {missing[:5]}",
                f"    unexpected in rebuild: {extra[:5]}",
            ]
        return []
    unknown = sorted(set(produced) - set(map(str, names)))
    if unknown:
        return [f"rebuilt classes not present in class_names.json: {unknown[:5]}"]
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE, help="merged split to rebuild (read only)")
    parser.add_argument("--out", type=Path, default=None, help="clean split to write (default data/merged_clean)")
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE, help="where dropped files are kept")
    parser.add_argument("--scope", default="all", help="'all', or a substring of class names such as 'wheat'")
    parser.add_argument("--apply", action="store_true", help="actually write the clean split (default: report only)")
    parser.add_argument("--force", action="store_true", help="write even if verification failed (you should not need this)")
    parser.add_argument("--mode", choices=("copy", "link"), default="copy", help="'link' hardlinks to save disk")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="share of *source photos* held out")
    parser.add_argument("--seed", type=int, default=42, help="same seed as merge_datasets.py")
    parser.add_argument("--min-val-per-class", type=int, default=10, help="promote train photos if a class falls below")
    parser.add_argument("--keep-mislabeled-val", action="store_true", help="do not quarantine val/Wheat___healthy")
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE, help="sha256 memo file")
    parser.add_argument("--class-names", type=Path, default=DEFAULT_CLASS_NAMES)
    parser.add_argument("--report", type=Path, default=HERE / "dataset_rebuild_report.txt")
    parser.add_argument("--csv", type=Path, default=HERE / "dataset_rebuild_manifest.csv")
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()
    args.out = args.out or (DEFAULT_OUT if args.scope == "all" else HERE / "data" / f"merged_clean_{args.scope}")
    args.mislabeled_val_classes = set() if args.keep_mislabeled_val else set(DEFAULT_MISLABELED_VAL_CLASSES)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 118)
    emit("AgroVeyra split rebuild - leak-free train/val from a duplicated, partly mislabeled merge")
    emit("=" * 118)
    emit(f"source       : {args.source}")
    emit(f"output       : {args.out}{'' if args.apply else '   (dry run: not written)'}")
    emit(f"quarantine   : {args.quarantine}")
    emit(f"scope        : {args.scope}")
    emit(f"write mode   : {args.mode}")
    emit(f"val fraction : {args.val_fraction} of source photos, split along photo boundaries")
    emit(f"mislabeled   : {', '.join(sorted(args.mislabeled_val_classes)) or 'none quarantined'}")
    emit(f"hash cache   : {args.cache}")
    emit("")

    if not args.source.is_dir():
        emit(f"source dataset not found: {args.source}")
        return 2

    cache = load_hash_cache(args.cache)
    dirty = [False]
    train_by_class = walk_split(args.source / "train", args.scope)
    val_by_class = walk_split(args.source / "val", args.scope)
    classes = sorted(set(train_by_class) | set(val_by_class))
    if not classes:
        emit(f"no images found under {args.source} for scope {args.scope!r}")
        return 2

    total_files = sum(len(files) for files in train_by_class.values()) + sum(len(files) for files in val_by_class.values())
    emit(f"classes in scope : {len(classes)}")
    emit(f"images in scope  : {total_files:,} (sha256 of each; cached after the first run)")
    emit("")

    records_by_class: dict[str, list[dict]] = {}
    for class_name in classes:
        records = collect_records(train_by_class.get(class_name, []), "train", class_name, args, cache, dirty)
        records += collect_records(val_by_class.get(class_name, []), "val", class_name, args, cache, dirty)
        records_by_class[class_name] = records
    if dirty[0]:
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        args.cache.write_text(json.dumps(cache), encoding="utf-8")
        emit(f"hash cache updated: {args.cache}")

    plan = build_plan(classes, records_by_class, args)
    rows = plan_destinations(plan, args)
    failures, stats = verify_plan(plan, args)
    order_failures = verify_class_order(classes, args.class_names, args.scope == "all")

    reassigned = sum(1 for row in rows if row["action"] in ("train", "val") and row["from_split"] != row["action"])
    promoted = sum(entry["promoted"] for entry in plan.values())

    emit("")
    emit("=" * 118)
    emit("1. PER-CLASS RESULT (before -> after)")
    emit("=" * 118)
    emit(f"  {'class':<40}{'train':>16}{'val':>16}{'quarantined':>14}")
    emit(f"  {'':<40}{'-' * 48}")
    for class_name in classes:
        before_train = len(train_by_class.get(class_name, []))
        before_val = len(val_by_class.get(class_name, []))
        after_train = len(plan[class_name]["splits"]["train"])
        after_val = len(plan[class_name]["splits"]["val"])
        gone = len(plan[class_name]["quarantined"])
        flag = "  <-- rebuilt to empty" if (after_train and not after_val) else ""
        emit(
            f"  {class_name:<40}"
            f"{f'{before_train:,} -> {after_train:,}':>16}"
            f"{f'{before_val:,} -> {after_val:,}':>16}"
            f"{gone:>14}{flag}"
        )
    emit("")
    emit(f"  totals: train {stats['train']:,} images, val {stats['val']:,} images, "
         f"{stats['quarantined']:,} quarantined, {reassigned:,} files changed side of the split")
    emit("")
    by_reason: dict[str, int] = {}
    for row in rows:
        if row["action"] == "quarantine":
            by_reason[row["reason"]] = by_reason.get(row["reason"], 0) + 1

    emit("=" * 118)
    emit("2. QUARANTINE BREAKDOWN (files are moved here, never deleted)")
    emit("=" * 118)
    labels = {
        REASON_CROSS_CLASS: "identical pixels under two different labels - unlearnable",
        REASON_SAME_CLASS: "same image stored twice inside one class",
        REASON_MISLABELED: "val/Wheat___healthy: source says healthy, the trained model says yellow_rust",
    }
    for reason, count in sorted(by_reason.items(), key=lambda item: -item[1]):
        emit(f"  {reason:<34}{count:>8}   {labels.get(reason, '')}")
    emit(f"  {'TOTAL':<34}{stats['quarantined']:>8}")
    emit(f"  source photos promoted train -> val to keep a class present: {promoted}")
    emit("")

    emit("=" * 118)
    emit("3. VERIFICATION")
    emit("=" * 118)
    emit(f"  identical images left on both sides of any class split : {stats['leaked_digests']}")
    emit("        (before: 486 unique digests leaked in wheat alone, 693 val files byte-identical to train)")
    emit(f"  source photos still spanning train and val             : {stats['shared_sources']}")
    emit("        (before: 450/1050 wheat val images, i.e. 42.9% of the wheat val set was already seen)")
    emit(f"  classes with a non-empty train and val split           : {len(classes) - len([f for f in failures if 'empty' in f])}/{len(classes)}")
    if order_failures:
        emit(f"  class_names.json contract                              : FAIL")
        for line in order_failures:
            emit(f"      {line}")
    else:
        emit(f"  class_names.json contract                              : OK")
        emit("        (same class set, same order - so the softmax index -> label mapping is unchanged)")

    ok = not failures and not order_failures
    if args.apply and (ok or args.force):
        emit("")
        emit(f"  writing {len(rows):,} files to {args.out} and {args.quarantine} ({args.mode}) ...")
        written = apply_plan(rows, args)
        emit(f"  wrote {written:,} files")
    elif args.apply:
        emit("")
        emit("  REFUSING TO WRITE: the plan did not pass verification (see the problems above).")
        emit("      Nothing was changed. Fix the dataset or the class list, then re-run.")
        emit("      --force overrides this, but writing a rejected plan gives you a broken training set.")

    emit("")
    emit("=" * 118)
    emit(f"VERDICT: {'leak-free rebuild, safe to train on' if ok else 'REBUILD REJECTED - see problems above'}")
    emit("=" * 118)
    if failures:
        emit("  problems:")
        for line in failures[:20]:
            emit(f"    - {line}")
        if len(failures) > 20:
            emit(f"    ... and {len(failures) - 20} more")
    if ok:
        if args.apply:
            emit(f"  next: train on {args.out} (never on {args.source}, which is still leaky):")
            emit("      ml/.venv/Scripts/python.exe ml/train_disease.py --data "
                 f"{args.out.relative_to(PROJECT_ROOT) if args.out.is_relative_to(PROJECT_ROOT) else args.out} "
                 "--name disease_clean")
        else:
            emit(f"  next: re-run with --apply to build {args.out}.")
            emit("      reporting is read-only; nothing has been written yet.")
        if args.scope != "all":
            emit(f"  NOTE: scope was {args.scope!r}, so this only covers part of the dataset.")
            emit("        Do not train on a partial rebuild; use --scope all for the real split.")
    emit("")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["action", "reason", "class", "from_split", "original_path", "destination_path"])
        for row in rows:
            writer.writerow(
                [
                    row["action"],
                    row["reason"],
                    row["class"],
                    row["from_split"],
                    str(row["original"]),
                    str(row["destination"]),
                ]
            )
    print(f"report   : {args.report}", flush=True)
    print(f"manifest : {args.csv}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
