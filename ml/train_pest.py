"""Train the AgroVeyra pest classifier (YOLO11n-cls) on the scoped IP102 subset.

Same architecture and same pipeline as the disease model (ml/train_disease.py), different task: 41
pest species that attack crops the app already diagnoses. Scope and evidence:
ml/pest_scope_report.txt, ml/pest_scoped_audit.txt, ml/pest_neardup_verification.txt.

The index contract matters here as much as it does for the disease model: Ultralytics derives class
indices by *sorting the class-folder names*, so ml/data/pest/ip102_scoped/pest_class_names.json must
equal sorted(folder names) or every prediction is the wrong species. This script refuses to train
otherwise, and re-checks the trained model's own names against the json afterwards.

Usage:
    ml\\.venv\\Scripts\\python.exe ml\\train_pest.py --env-only
    ml\\.venv\\Scripts\\python.exe ml\\train_pest.py
    ml\\.venv\\Scripts\\python.exe ml\\train_pest.py --batch 8 --epochs 50 --name pest_classifier_scoped
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Same guidance as the disease run: YOLO11n-cls at 224 px, by total VRAM.
VRAM_BATCH_TABLE = (
    (4.0, 8, "consider --imgsz 160 as well"),
    (6.0, 16, "the RTX 3050 6GB runs the disease model at batch 16"),
    (8.0, 32, ""),
    (12.0, 64, ""),
    (float("inf"), 96, ""),
)


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def preflight() -> tuple[bool, float]:
    """Print the environment report; return (cuda_ready, vram_gb)."""
    print("=" * 96)
    print("Environment preflight")
    print("=" * 96)
    try:
        import ultralytics
    except ImportError:
        print("  ultralytics : NOT INSTALLED")
        return False, 0.0
    print(f"  python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"  ultralytics : {ultralytics.__version__}")
    try:
        import torch
    except ImportError:
        print("  torch       : NOT INSTALLED")
        return False, 0.0
    print(f"  torch       : {torch.__version__}")
    print(f"  torch cuda  : {torch.version.cuda or 'not a CUDA build'}")
    if not torch.cuda.is_available():
        print("  cuda ready  : False - train on a CUDA host; CPU days are not worth it here")
        return False, 0.0
    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    print(f"  gpu         : {properties.name} ({vram_gb:.1f} GB)")
    print("  cuda ready  : True")
    return True, vram_gb


def suggest_batch(vram_gb: float) -> tuple[int, str]:
    for limit, batch, note in VRAM_BATCH_TABLE:
        if vram_gb <= limit:
            return batch, note
    return 16, ""


def class_contract(data_dir: Path) -> tuple[bool, list[str], list[str], dict]:
    """Check pest_class_names.json against the sorted folder names of train/ and val/."""
    json_path = data_dir / "pest_class_names.json"
    report = {"json": json_path.name, "counts": {}}
    if not json_path.is_file():
        print(f"  class contract : MISSING {json_path}")
        return False, [], [], report
    names = json.loads(json_path.read_text(encoding="utf-8"))
    folders = {}
    for split in ("train", "val", "test"):
        split_dir = data_dir / split
        if split_dir.is_dir():
            folders[split] = sorted(d.name for d in split_dir.iterdir() if d.is_dir())
            report["counts"][split] = len(folders[split])
    expected = folders.get("train", [])
    ok = names == expected
    if not ok:
        print(f"  class contract : FAIL - json has {len(names)} names, train has {len(expected)} folders")
        for i, (a, b) in enumerate(zip(names, expected)):
            if a != b:
                print(f"    first mismatch at index {i}: json={a!r} folder={b!r}")
                break
    for split, listing in folders.items():
        if listing and listing != expected:
            print(f"  class contract : FAIL - {split} folders differ from train")
            ok = False
    return ok, names, expected, report


def parse_args():
    parser = argparse.ArgumentParser(description="Train the scoped pest classifier.")
    parser.add_argument("--data", default="data/pest/ip102_scoped")
    parser.add_argument("--model", default="yolo11n-cls.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="pest_classifier_scoped")
    parser.add_argument("--project", default="runs")
    parser.add_argument("--env-only", action="store_true")
    parser.add_argument("--check-only", action="store_true",
                        help="preflight + dataset/contract checks, then stop before training")
    parser.add_argument("--skip-contract-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    cuda_ready, vram_gb = preflight()
    if args.env_only:
        return 0 if cuda_ready else 1

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (HERE / data_dir).resolve()
    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = (HERE / project_dir).resolve()

    data_ready = (data_dir / "train").is_dir() and (data_dir / "val").is_dir()
    print("")
    print("Dataset")
    print("=" * 96)
    if not data_ready:
        print(f"  dataset not found: {data_dir}")
        print("  build it first: ml\\.venv\\Scripts\\python.exe ml\\build_pest_scoped.py")
        return 2

    if args.skip_contract_check:
        print("  class contract : SKIPPED - index -> species mapping unverified")
    else:
        ok, names, folders, report = class_contract(data_dir)
        print(f"  splits         : " + ", ".join(f"{k}={v}" for k, v in report["counts"].items()))
        if not ok:
            print("  refusing to train: pest_class_names.json disagrees with the folder order.")
            return 3
        print(f"  class contract : OK - {len(names)} classes, json equals sorted(folder names)")
        print(f"  index 0        : {names[0]}")
        print(f"  index {len(names) - 1:<8}: {names[-1]}")

    suggested, note = suggest_batch(vram_gb)
    if str(args.batch).lower() == "auto":
        batch: int | str = -1
        print(f"  batch          : -1 (auto from free VRAM; expect about {suggested}"
              f"{f' - {note}' if note else ''})")
    else:
        batch = int(args.batch)
        print(f"  batch          : {batch} (explicit)")
    print(f"  project        : {project_dir}")
    print(f"  run name       : {args.name}")
    print("=" * 96)
    print("")

    if args.check_only:
        print("--check-only: environment, dataset and class contract verified. Not training.")
        return 0

    from ultralytics import YOLO

    model = YOLO(args.model)
    results = model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(project_dir),
        name=args.name,
        seed=args.seed,
    )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print("")
    print(f"Training complete. Results saved to: {save_dir}")

    trained_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    expected = json.loads((data_dir / "pest_class_names.json").read_text(encoding="utf-8"))
    print(f"  model.names matches pest_class_names.json: {trained_names == expected}")
    if trained_names != expected:
        print("  WARNING: index order differs from the json - fix the json before shipping the app")
        for i, (a, b) in enumerate(zip(trained_names, expected)):
            if a != b:
                print(f"    first mismatch at index {i}: model={a!r} json={b!r}")
                break

    try:
        val_metrics = model.val(data=str(data_dir))
        print(f"  val top-1/top-5 : {val_metrics.top1:.4f} / {val_metrics.top5:.4f}")
    except Exception as error:
        print(f"  val metrics unavailable: {error}")

    try:
        test_metrics = model.val(data=str(data_dir), split="test")
        print(f"  test top-1/top-5: {test_metrics.top1:.4f} / {test_metrics.top5:.4f}")
    except Exception as error:
        print(f"  test split unavailable ({error}); evaluate it separately for the honest number")

    print("")
    print("Next: export to TFLite, verify parity, then wire the asset into the app.")
    print(f"  checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
