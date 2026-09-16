#!/usr/bin/env python
"""Dataset integrity audit for `data/merged`: duplicates, label conflicts and train/val leakage.

`audit_datasets.py` audits the *structure* of the source datasets (folder trees, class counts,
naming patterns). This script audits their *content*: whether the merged split can be learned and
whether the validation metric means anything.

`diagnose_class_errors.py` says *where* the deployed classifier fails; this script checks whether
the data underneath it can be learned at all. Three defects are looked for:

  1. TRAIN/VAL OVERLAP       val images that are not really held out, measured two ways:
                             (a) same *source photo* as a training image - the merge step wrote
                                 offline rotations/flips (`..._270deg.JPG`, `..._FlipLR.JPG`) as
                                 separate files and then split them across train/val;
                             (b) byte-identical files on both sides of the split
  2. SAME-CLASS DUPLICATES   the same file kept twice inside one split (skews per-class counts)
  3. CROSS-CLASS DUPLICATES  identical pixels carrying two different labels - unlearnable, and the
                             usual cause of persistent confusion between sibling classes

Hashing every file in scope takes minutes, so `--scope` narrows the audit to one crop family; that is
enough to act on, because the errors are not uniformly spread across crops.

Usage:
    ml/.venv/Scripts/python.exe ml/audit_dataset_integrity.py --scope wheat
    ml/.venv/Scripts/python.exe ml/audit_dataset_integrity.py --scope all     # slow, whole data dir
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = HERE / "data" / "merged"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# "<uuid>___<name>_270deg.JPG": the merge step wrote offline rotations/flips as separate files.
SOURCE_SEPARATOR = re.compile(r"_{3,}")
UUID = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_key(path: Path) -> str:
    """Identity of the *source* photo, ignoring the augmentation suffix.

    `00075aa8-...___FREC_Scab 3335_270deg.JPG` and `00075aa8-...___FREC_Scab 3335.JPG` are the same
    leaf photographed once, so they must not sit on opposite sides of the split.
    """
    prefix = SOURCE_SEPARATOR.split(path.stem, maxsplit=1)[0]
    match = UUID.search(prefix)
    return (match.group(0) if match else prefix).lower()


def walk(directory: Path, scope: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return [
        path
        for path in directory.rglob("*")
        if path.suffix.lower() in IMAGE_SUFFIXES and (scope == "all" or scope in path.parent.name.lower())
    ]


def digest_index(paths: list[Path]) -> dict[str, list[Path]]:
    """sha256 -> files.

    Every file is hashed rather than only same-named twins: cross-class duplicates *do* carry
    different file names (`aphid_130.png` == `leaf_blight_519.png`), so a name-based pre-filter
    silently hides exactly the defect this audit is looking for. Hashing the whole scope is the
    price, which is why `--scope` exists.
    """
    index: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        index[sha256(path)].append(path)
    return index


def main() -> int:
    configure_stdout()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--scope",
        default="all",
        help="'all', or a substring of the class folder names to restrict the audit to (e.g. wheat)",
    )
    parser.add_argument("--examples", type=int, default=10, help="example rows printed per section")
    parser.add_argument("--report", type=Path, default=HERE / "dataset_integrity_report.txt")
    parser.add_argument("--csv", type=Path, default=HERE / "dataset_integrity_conflicts.csv")
    args = parser.parse_args()

    scope = args.scope.lower()
    train_files = walk(args.data_dir / "train", scope)
    val_files = walk(args.data_dir / "val", scope)
    if not train_files and not val_files:
        raise SystemExit(f"nothing to audit under {args.data_dir} for scope {args.scope!r}")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit("=" * 118)
    emit("AgroVeyra dataset integrity audit")
    emit("=" * 118)
    emit(f"data dir      : {args.data_dir}")
    emit(f"scope         : {args.scope}")
    emit(f"train files   : {len(train_files)}")
    emit(f"val files     : {len(val_files)}")
    emit("")

    emit("=" * 118)
    emit("1. TRAIN/VAL OVERLAP")
    emit("=" * 118)
    train_sources = {source_key(path) for path in train_files}
    leaked_val = [(path, source_key(path)) for path in val_files if source_key(path) in train_sources]
    emit("  1a. source-image overlap (offline rotations/flips of one photo split across the sets)")
    emit(f"      val images whose source photo also appears in train: {len(leaked_val)}/{len(val_files)} "
         f"({len(leaked_val) / len(val_files) * 100:.1f}%)")
    per_class: dict[str, int] = defaultdict(int)
    for path, _key in leaked_val:
        per_class[path.parent.name] += 1
    for name, count in sorted(per_class.items(), key=lambda item: -item[1])[: args.examples]:
        total = sum(1 for path in val_files if path.parent.name == name)
        emit(f"        {count:>5}/{total:<5} {name}")
    for path, key in leaked_val[: args.examples]:
        emit(f"        {path.relative_to(args.data_dir)}  (source {key[:8]})")
    emit("")

    train_digests = digest_index(train_files)
    val_digests = digest_index(val_files)

    leakage = [
        (digest, val_digests[digest], train_digests[digest])
        for digest in val_digests.keys() & train_digests.keys()
    ]
    emit("  1b. byte-identical overlap (the strictest possible test)")
    emit(f"      leaked val images : {sum(len(val) for _, val, _ in leakage)}")
    emit(f"      unique digests    : {len(leakage)}")
    for _digest, val_paths, train_paths in leakage[: args.examples]:
        emit(f"        {val_paths[0].relative_to(args.data_dir)} == {train_paths[0].relative_to(args.data_dir)}")
    emit("")

    train_dupes = [paths for paths in train_digests.values() if len(paths) > 1]
    val_dupes = [paths for paths in val_digests.values() if len(paths) > 1]
    emit("=" * 118)
    emit("2. SAME-CLASS DUPLICATES (same image stored twice in one split)")
    emit("=" * 118)
    emit(f"  train duplicate groups: {len(train_dupes)} ({sum(len(paths) - 1 for paths in train_dupes)} redundant files)")
    emit(f"  val duplicate groups  : {len(val_dupes)} ({sum(len(paths) - 1 for paths in val_dupes)} redundant files)")
    for paths in (train_dupes + val_dupes)[: args.examples]:
        emit(f"    {', '.join(str(path.relative_to(args.data_dir)) for path in paths[:4])}")
    emit("")

    conflicts = [paths for paths in train_dupes if len({path.parent.name for path in paths}) > 1]
    emit("=" * 118)
    emit("3. CROSS-CLASS DUPLICATES (identical pixels, different labels - unlearnable)")
    emit("=" * 118)
    emit(f"  conflicting training groups: {len(conflicts)}")
    emit(f"  affected files             : {sum(len(paths) for paths in conflicts)}")
    for paths in sorted(conflicts, key=lambda group: -len(group))[: args.examples]:
        emit(f"    {' , '.join(str(path.relative_to(args.data_dir)) for path in paths[:4])}")
    emit("")

    pairs: dict[tuple[str, str], int] = defaultdict(int)
    for paths in conflicts:
        classes = sorted({path.parent.name for path in paths})
        for left_index, left in enumerate(classes):
            for right in classes[left_index + 1 :]:
                pairs[(left, right)] += 1
    if pairs:
        emit("  label pairs that share pixels (worst first)")
        for (left, right), count in sorted(pairs.items(), key=lambda item: -item[1])[: args.examples]:
            emit(f"    {count:>5} shared images  {left}  <->  {right}")
        emit("")

    verdict = not leakage and not conflicts and not leaked_val
    emit("=" * 118)
    emit(f"VERDICT: {'clean' if verdict else 'defects found'}")
    emit("=" * 118)
    if not verdict:
        emit("  a source-image overlap means the val split is not held out (the metric is optimistic),")
        emit("  and every cross-class duplicate is a training example that cannot be learned correctly.")
    emit("")

    args.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with args.csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "label_a", "label_b", "file_a", "file_b"])
        for _digest, val_paths, train_paths in leakage:
            writer.writerow(
                [
                    "train_val_overlap",
                    val_paths[0].parent.name,
                    train_paths[0].parent.name,
                    str(val_paths[0]),
                    str(train_paths[0]),
                ]
            )
        for paths in conflicts:
            writer.writerow(
                [
                    "cross_class_duplicate",
                    paths[0].parent.name,
                    paths[1].parent.name,
                    str(paths[0]),
                    str(paths[1]),
                ]
            )
    print(f"report: {args.report}", flush=True)
    print(f"csv   : {args.csv}", flush=True)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

