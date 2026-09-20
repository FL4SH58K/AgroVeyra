"""Guard the shipping class contract (60 classes across 17 crops) before training or exporting.

The class list is a *contract*, not an implementation detail: `backend/models/class_names.json`
defines the softmax index -> label mapping that `TFLiteHelper.kt` and `backend/utils/predict.py`
depend on, and `android/app/src/main/assets/class_names.json` must stay a byte-identical copy of it.
Nothing in the training path used to check that, so a dataset could silently disagree with the app
(missing class, renamed class, reordered list) and the model would still train and predict the wrong
crop with full confidence.

`GROUP_5_PROJECT.pdf` asks for "52 disease categories across 17 crop species". The 52 is a
documentation figure only - the shipped implementation target is the existing 60-class set - so this
guard pins the *current* contract and fails loudly on any drift in either direction.

Read-only. Exits non-zero on the first violated gate. It asserts:

  1. class_names.json parses, holds 60 unique non-empty names, and is already in Python `sorted()`
     order (that order *is* the TFLite output index order)
  2. those names still cover 17 crop species (PlantVillage's three double-spellings merged)
  3. the Android asset is byte-identical to the backend copy
  4. all 60 classes have a treatment entry, and both treatment_db.json copies are identical
  5. the shipped TFLite (backend copy == Android copy) matches the size/sha256 recorded in
     ml/export_parity_report.txt, and that report says `classes : 60`
  6. a dataset split (default data/merged_clean, falling back to data/merged) has exactly these 60
     class folders in both train/ and val/, in the same order

Usage:
    ml/.venv/Scripts/python.exe ml/verify_class_contract.py
    ml/.venv/Scripts/python.exe ml/verify_class_contract.py --data ml/data/merged
    ml/.venv/Scripts/python.exe ml/verify_class_contract.py --no-dataset
    ml/.venv/Scripts/python.exe ml/verify_class_contract.py --json ml/class_contract_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_CONTRACT = PROJECT_ROOT / "backend" / "models" / "class_names.json"
DEFAULT_ANDROID_ASSET = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "class_names.json"
DEFAULT_TREATMENT = PROJECT_ROOT / "backend" / "treatment_db.json"
DEFAULT_ANDROID_TREATMENT = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "treatment_db.json"
DEFAULT_TFLITE = PROJECT_ROOT / "backend" / "models" / "agroveyra_model.tflite"
DEFAULT_ANDROID_TFLITE = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "agroveyra_model.tflite"
DEFAULT_PARITY_REPORT = HERE / "export_parity_report.txt"
DEFAULT_DATA_CLEAN = HERE / "data" / "merged_clean"
DEFAULT_DATA_MERGED = HERE / "data" / "merged"

EXPECTED_CLASSES = 60
EXPECTED_CROPS = 17

# PlantVillage ships three crops under two spellings each. Merging the spellings is what turns the
# 20 raw `Crop___Disease` prefixes into the 17 crop species the app and the report advertise.
CROP_ALIASES = {
    "Cherry_(including_sour)": "Cherry",
    "Corn_(maize)": "Corn",
    "Pepper,_bell": "Pepper",
}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_of(class_name: str) -> str:
    """Crop species behind a `Crop___Disease` folder name, with the PlantVillage spellings merged."""
    crop = class_name.split("___", 1)[0].strip()
    return CROP_ALIASES.get(crop, crop)


def parse_parity_report(path: Path) -> dict:
    """Fingerprint recorded by ml/verify_export_parity.py for the shipped flatbuffer (best effort)."""
    info: dict[str, object] = {}
    if not path.is_file():
        return info
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("flatbuffer"):
            for follow in lines[index + 1:index + 3]:
                match = re.search(r"([\d,]+) bytes \(sha256 ([0-9a-fA-F]+)\)", follow)
                if match:
                    info["bytes"] = int(match.group(1).replace(",", ""))
                    info["sha256"] = match.group(2).lower()
                    break
        if stripped.startswith("classes"):
            match = re.search(r"classes\s*:\s*(\d+)", stripped)
            if match:
                info["classes"] = int(match.group(1))
    return info


def main(argv: list[str] | None = None) -> int:
    configure_stdout()
    args = parse_args(argv)

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str = "") -> bool:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})
        emit(f"  {'PASS' if ok else 'FAIL'}  {name}{f' - {detail}' if detail else ''}")
        return bool(ok)

    args.data_was_default = args.data is None
    if args.data is None:
        args.data = DEFAULT_DATA_CLEAN if DEFAULT_DATA_CLEAN.is_dir() else DEFAULT_DATA_MERGED

    emit("=" * 96)
    emit("AgroVeyra class contract - 60 classes across 17 crops (read-only guard)")
    emit("=" * 96)
    emit(f"  contract     : {args.contract}")
    emit(f"  expecting    : {args.expect_count} classes, {args.expect_crops} crops")
    emit(f"  dataset      : {args.data}{' (default)' if args.data_was_default else ''}")
    emit("")

    # --- 1. the class list itself -------------------------------------------------------------
    emit("1. CLASS LIST (the softmax index -> label contract)")
    contract: list[str] = []
    if not args.contract.is_file():
        record("contract file exists", False, str(args.contract))
    else:
        try:
            payload = json.loads(args.contract.read_text(encoding="utf-8-sig"))
        except Exception as error:
            payload = None
            record("contract file parses as JSON", False, repr(error))
        if isinstance(payload, list):
            contract = [str(name) for name in payload]
            record("contract file parses as JSON", True, f"list of {len(contract)} names")
        elif payload is not None:
            record("contract file parses as JSON", False, f"top level is {type(payload).__name__}, expected a list")

    if contract:
        blanks = [name for name in contract if not name.strip()]
        duplicates = sorted({name for name in contract if contract.count(name) > 1})
        unpadded = [name for name in contract if name != name.strip()]
        record("no blank class names", not blanks, f"{len(blanks)} blank")
        record("no duplicate class names", not duplicates, ", ".join(duplicates[:3]))
        record("no stray whitespace in names", not unpadded, ", ".join(repr(name) for name in unpadded[:3]))
        record(
            f"class count is {args.expect_count}",
            len(contract) == args.expect_count,
            f"found {len(contract)}",
        )
        ordered = sorted(contract)
        mismatch = next(
            (
                f"index {index}: {contract[index]!r} vs sorted {ordered[index]!r}"
                for index in range(min(len(contract), len(ordered)))
                if contract[index] != ordered[index]
            ),
            "",
        )
        record(
            "already in Python sorted() order (is the TFLite index order)",
            contract == ordered,
            mismatch,
        )

    if contract and args.expect_crops > 0:
        crops = sorted({crop_of(name) for name in contract})
        prefixes = {name.split("___", 1)[0] for name in contract}
        record(
            f"crop coverage is {args.expect_crops} species",
            len(crops) == args.expect_crops,
            f"found {len(crops)}: {', '.join(crops)}",
        )
        emit(f"        {len(contract)} classes -> {len(crops)} crops ({len(prefixes)} raw prefixes; "
             "PlantVillage double-spellings merged)")

    # --- 2. Android ships the same file -------------------------------------------------------
    emit("")
    emit("2. ANDROID ASSET (must be a byte-identical copy)")
    if not args.android.is_file():
        record("android class_names.json exists", False, str(args.android))
    elif contract:
        backend_digest = sha256_of(args.contract)
        android_digest = sha256_of(args.android)
        record(
            "android class_names.json == backend class_names.json",
            backend_digest == android_digest,
            f"sha256 {backend_digest[:16]} vs {android_digest[:16]}",
        )

    # --- 3. every class keeps its treatment ---------------------------------------------------
    emit("")
    emit("3. TREATMENT DATABASE (every class must have advisory text)")
    treatment: dict = {}
    if not args.treatment.is_file():
        record("treatment_db.json exists", False, str(args.treatment))
    else:
        try:
            loaded = json.loads(args.treatment.read_text(encoding="utf-8-sig"))
        except Exception as error:
            loaded = None
            record("treatment_db.json parses as JSON", False, repr(error))
        if isinstance(loaded, dict):
            treatment = loaded
            record("treatment_db.json parses as JSON", True, f"{len(treatment)} keys")
        elif loaded is not None:
            record("treatment_db.json parses as JSON", False,
                   f"top level is {type(loaded).__name__}, expected an object")

    if contract and treatment:
        missing = [name for name in contract if name not in treatment]
        extras = sorted(set(treatment) - set(contract))
        record(
            "every class has a treatment entry",
            not missing,
            f"{len(missing)} missing: {', '.join(missing[:3])}" if missing else "",
        )
        emit(f"        {len(treatment)} entries = {len(contract) - len(missing)} model classes "
             f"+ {len(extras)} extra key(s)")
        if not args.android_treatment.is_file():
            record("android treatment_db.json exists", False, str(args.android_treatment))
        else:
            record(
                "android treatment_db.json == backend treatment_db.json",
                sha256_of(args.treatment) == sha256_of(args.android_treatment),
            )

    # --- 4. the flatbuffer that actually ships ------------------------------------------------
    emit("")
    emit("4. SHIPPED FLATBUFFER (fingerprint recorded by verify_export_parity.py)")
    parity = parse_parity_report(args.parity_report)
    if not args.tflite.is_file():
        record("backend agroveyra_model.tflite exists", False, str(args.tflite))
    elif not args.android_tflite.is_file():
        record("android agroveyra_model.tflite exists", False, str(args.android_tflite))
    else:
        digest = sha256_of(args.tflite)
        record("android tflite == backend tflite", digest == sha256_of(args.android_tflite))
        emit(f"        {args.tflite.stat().st_size:,} bytes (sha256 {digest[:16]})")
        recorded = str(parity.get("sha256") or "")
        if recorded:
            record(
                "matches the fingerprint in export_parity_report.txt",
                digest.startswith(recorded),
                f"on disk {digest[:16]}, report {recorded[:16]}",
            )
        else:
            emit(f"        note: no flatbuffer fingerprint in {args.parity_report.name} - not compared")
        if "classes" in parity:
            record(
                f"parity report says {args.expect_count} classes",
                parity["classes"] == args.expect_count,
                f"report says {parity['classes']}",
            )

    # --- 5. the dataset folders the trainer reads ---------------------------------------------
    emit("")
    emit("5. DATASET (folder names are the training labels)")
    if args.no_dataset:
        emit("  SKIP  dataset check disabled (--no-dataset)")
    elif not args.data.is_dir():
        record("dataset split exists", False, f"{args.data} (build it first: ml/GPU_RUNBOOK.md step 2)")
    elif not contract:
        record("dataset checked against the contract", False, "no class list to compare against")
    else:
        expected = sorted(contract)
        for split in ("train", "val"):
            split_dir = args.data / split
            if not split_dir.is_dir():
                record(f"{split}/ exists", False, str(split_dir))
                continue
            folders = sorted(entry.name for entry in split_dir.iterdir() if entry.is_dir())
            missing = [name for name in expected if name not in folders]
            extra = [name for name in folders if name not in expected]
            detail = f"{len(folders)} folders"
            if missing:
                detail += f", missing {len(missing)}"
            if extra:
                detail += f", unexpected {len(extra)}"
            record(
                f"{split}/ holds exactly the {args.expect_count} contract classes",
                not missing and not extra,
                detail,
            )
            if missing:
                emit(f"        missing: {', '.join(missing[:5])}")
            if extra:
                emit(f"        unexpected: {', '.join(extra[:5])}")
            counts = {
                name: sum(
                    1
                    for file in (split_dir / name).iterdir()
                    if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES
                )
                for name in folders
            }
            empty = sorted(name for name, count in counts.items() if count == 0)
            record(f"{split}/ has no empty class folder", not empty,
                   f"{len(empty)} empty: {', '.join(empty[:3])}")
            if counts:
                smallest = sorted(counts.items(), key=lambda item: (item[1], item[0]))[:3]
                emit(f"        {sum(counts.values()):,} images; smallest classes: "
                     + ", ".join(f"{name} ({count:,})" for name, count in smallest))

    failed = [check for check in checks if not check["ok"]]
    emit("")
    emit("=" * 96)
    if failed:
        emit("VERDICT: CONTRACT VIOLATED - do not train or ship this state")
    else:
        emit("VERDICT: contract holds - safe to train and ship")
    emit("=" * 96)
    emit(f"  checks passed : {len(checks) - len(failed)}/{len(checks)}")
    if failed:
        emit("  violations:")
        for check in failed:
            detail = f" ({check['detail']})" if check["detail"] else ""
            emit(f"    - {check['name']}{detail}")
    emit("")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "ok": not failed,
                    "contract": str(args.contract),
                    "android_asset": str(args.android),
                    "dataset": str(args.data),
                    "expect_count": args.expect_count,
                    "expect_crops": args.expect_crops,
                    "checks": checks,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"report : {args.json}")
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT, help="class_names.json (the contract)")
    parser.add_argument("--android", type=Path, default=DEFAULT_ANDROID_ASSET, help="Android copy of the contract")
    parser.add_argument("--treatment", type=Path, default=DEFAULT_TREATMENT)
    parser.add_argument("--android-treatment", type=Path, default=DEFAULT_ANDROID_TREATMENT)
    parser.add_argument("--tflite", type=Path, default=DEFAULT_TFLITE)
    parser.add_argument("--android-tflite", type=Path, default=DEFAULT_ANDROID_TFLITE)
    parser.add_argument("--parity-report", type=Path, default=DEFAULT_PARITY_REPORT)
    parser.add_argument("--data", type=Path, default=None,
                        help="dataset split to check (default: data/merged_clean, else data/merged)")
    parser.add_argument("--no-dataset", action="store_true", help="skip the dataset folder check")
    parser.add_argument("--expect-count", type=int, default=EXPECTED_CLASSES, help="required number of classes")
    parser.add_argument("--expect-crops", type=int, default=EXPECTED_CROPS,
                        help="required number of crop species (0 disables)")
    parser.add_argument("--json", type=Path, default=None, help="write a machine-readable summary here")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
