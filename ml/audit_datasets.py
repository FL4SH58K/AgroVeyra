from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


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

SPLIT_NAMES = {"train", "val", "valid", "test"}


@dataclass
class DatasetReport:
    name: str
    root: Path
    exists: bool
    total_images: int = 0
    class_count: int = 0
    split_paths: list[Path] = field(default_factory=list)
    class_pattern: str = "unknown"
    warnings: list[str] = field(default_factory=list)


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def count_direct_images(folder: Path) -> int:
    return sum(1 for entry in folder.iterdir() if is_image_file(entry))


def count_recursive_images(folder: Path) -> int:
    return sum(1 for entry in folder.rglob("*") if is_image_file(entry))


def format_indent(depth: int) -> str:
    return "    " * depth


def scan_tree(folder: Path, depth: int = 0, max_depth: int = 3) -> list[str]:
    lines: list[str] = []
    if depth > max_depth:
        return lines
    if not folder.is_dir():
        return lines

    try:
        children = sorted(folder.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
    except PermissionError:
        lines.append(f"{format_indent(depth)}[permission denied]")
        return lines

    dir_children = [child for child in children if child.is_dir()]
    image_children = [child for child in children if is_image_file(child)]

    prefix = format_indent(depth)
    label = folder.name + "/" if folder.name else str(folder)
    line = f"{prefix}{label}  [dirs={len(dir_children)}, images={len(image_children)}]"
    if len(image_children) == 0:
        line += "  !0img"
    lines.append(line)

    if depth == max_depth:
        if dir_children:
            lines.append(f"{prefix}    ... depth limit reached ({len(dir_children)} subfolders hidden)")
        return lines

    for child in dir_children:
        branch = scan_tree(child, depth + 1, max_depth)
        if branch:
            lines.extend(branch)
    return lines


def walk_dirs(folder: Path, depth: int = 0, max_depth: int = 3) -> Iterable[tuple[Path, int]]:
    yield folder, depth
    if depth >= max_depth:
        return
    try:
        for child in sorted(folder.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir():
                yield from walk_dirs(child, depth + 1, max_depth)
    except PermissionError:
        return


def detect_split_paths(root: Path) -> list[Path]:
    split_paths: list[Path] = []
    for folder, _depth in walk_dirs(root):
        if folder.name.lower() not in {"train", "test", "val", "valid"}:
            continue
        try:
            parent_names = {child.name.lower() for child in folder.parent.iterdir() if child.is_dir()}
        except PermissionError:
            continue
        has_train = folder.name.lower() == "train"
        has_val = "val" in parent_names
        has_valid = "valid" in parent_names
        has_test = "test" in parent_names
        if has_train and (has_val or has_valid):
            split_paths.append(folder.parent)
        elif folder.name.lower() in {"val", "valid"} and "train" in parent_names:
            split_paths.append(folder.parent)
        elif has_test and "train" in parent_names and (has_val or has_valid):
            split_paths.append(folder.parent)

    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in split_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)
    return unique_paths


def discover_class_dirs(root: Path, max_depth: int = 8) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for folder, _depth in walk_dirs(root, max_depth=max_depth):
        try:
            direct_images = count_direct_images(folder)
        except PermissionError:
            continue
        if direct_images == 0:
            continue
        resolved = folder.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        candidates.append(folder)
    return candidates


def normalize_class_label(folder: Path) -> str:
    name = folder.name.lower()
    suffixes = ("_train", "_test", "_val", "_valid")

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                changed = True
                break
    return name


def class_count_key(folder: Path) -> str:
    normalized = normalize_class_label(folder)
    return "".join(character for character in normalized if character.isalnum())


def classify_name_pattern(class_dirs: list[Path]) -> str:
    if not class_dirs:
        return "unknown"

    names = [normalize_class_label(folder) for folder in class_dirs]
    numeric = all(name.isdigit() for name in names)
    crop_disease = all("___" in name for name in names)

    if numeric:
        return "numeric_class_ids"
    if crop_disease:
        return "crop___disease"
    if any("___" in name for name in names):
        return "mixed (crop___disease + other)"
    return "plain_label_names"


def path_depth_from(root: Path, target: Path) -> int:
    try:
        return len(target.resolve().relative_to(root.resolve()).parts)
    except Exception:
        return -1


def collect_unusual_nesting(root: Path) -> list[str]:
    warnings: list[str] = []
    for folder, depth in walk_dirs(root):
        try:
            direct_images = count_direct_images(folder)
            dir_children = [child for child in folder.iterdir() if child.is_dir()]
        except PermissionError:
            continue

        if depth > 3:
            warnings.append(f"[depth>{3}] {folder}")
        if direct_images == 0 and not dir_children:
            warnings.append(f"[0 images] {folder}")
        if folder.name.lower() in SPLIT_NAMES and len(dir_children) == 1 and dir_children[0].name.lower() == folder.name.lower():
            warnings.append(f"[nested split folder] {folder}")
        if len(dir_children) == 1 and folder.name.lower() == dir_children[0].name.lower():
            warnings.append(f"[repeated folder name] {folder}")
    return warnings


def build_report(name: str, root: Path) -> DatasetReport:
    if not root.exists():
        return DatasetReport(name=name, root=root, exists=False)

    class_dirs = discover_class_dirs(root)
    unique_class_labels = sorted({class_count_key(folder) for folder in class_dirs})
    total_images = sum(count_recursive_images(folder) for folder in class_dirs)
    split_paths = detect_split_paths(root)
    warnings = collect_unusual_nesting(root)

    report = DatasetReport(
        name=name,
        root=root,
        exists=True,
        total_images=total_images,
        class_count=len(unique_class_labels),
        split_paths=split_paths,
        class_pattern=classify_name_pattern(class_dirs),
        warnings=warnings,
    )

    return report


def print_report(report: DatasetReport) -> None:
    print(f"\n=== {report.name} ===")
    print(f"Root: {report.root}")

    if not report.exists:
        print("NOT FOUND")
        print("Summary: total images = 0 | class folders = 0 | train/val split = no")
        return

    print("Folder tree (up to 3 levels):")
    for line in scan_tree(report.root):
        print(line)

    print("Class folders:")
    class_dirs = discover_class_dirs(report.root)
    if not class_dirs:
        print("  (none found)")
    else:
        for folder in class_dirs:
            image_count = count_recursive_images(folder)
            print(f"  - {folder.relative_to(report.root)} -> {image_count} images")

    print(f"Class naming pattern: {report.class_pattern}")
    if report.split_paths:
        print("Train/val split: yes")
        for path in report.split_paths:
            print(f"  split root: {path}")
    else:
        print("Train/val split: no")

    if report.warnings:
        print("Warnings:")
        for warning in report.warnings:
            print(f"  - {warning}")
    else:
        print("Warnings: none")

    print(
        f"Summary: total images = {report.total_images} | class count = {report.class_count} | train/val split = {'yes' if report.split_paths else 'no'}"
    )


def main() -> None:
    base = Path(__file__).resolve().parent
    datasets = [
        ("plantvillage", base / "data" / "disease" / "plantvillage"),
        ("rice", base / "data" / "disease" / "rice"),
        ("wheat", base / "data" / "disease" / "wheat"),
        ("cotton", base / "data" / "disease" / "cotton"),
        ("ip102", base / "data" / "pest" / "ip102"),
    ]

    reports = [build_report(name, root) for name, root in datasets]

    print("Dataset audit")
    print("=============")
    for report in reports:
        print_report(report)


if __name__ == "__main__":
    main()
