"""Train the AgroVeyra 3-way triage classifier (healthy / disease / pest_damage).

Third additive capability alongside the disease and pest models. yolo11n-cls at
224px, batch 16 (RTX 3050 6GB), 50 epochs, patience 10, seed 42.

Ultralytics sorts class-folder names, so the folders healthy/disease/pest_damage
sort to ["disease","healthy","pest_damage"], which triage_class_names.json records.
This script refuses to train unless the json equals sorted(folder names).

Usage:
    ml/.venv/Scripts/python.exe ml/train_triage.py --env-only
    ml/.venv/Scripts/python.exe ml/train_triage.py --name pest_triage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

VRAM_BATCH_TABLE = (
    (4.0, 8, "consider --imgsz 160 as well"),
    (6.0, 16, "the RTX 3050 6GB default"),
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
        print("  cuda ready  : False - train on a CUDA host")
        return False, 0.0
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024**3
    print(f"  gpu         : {props.name} ({vram_gb:.1f} GB)")
    print("  cuda ready  : True")
    return True, vram_gb


def suggest_batch(vram_gb: float) -> tuple[int, str]:
    for limit, batch, note in VRAM_BATCH_TABLE:
        if vram_gb <= limit:
            return batch, note
    return 16, ""


def class_contract(data_dir: Path) -> tuple[bool, list[str]]:
    json_path = data_dir / "triage_class_names.json"
    if not json_path.is_file():
        print(f"  class contract : MISSING {json_path}")
        return False, []
    names = json.loads(json_path.read_text(encoding="utf-8"))
    folders = {}
    for split in ("train", "val"):
        d = data_dir / split
        folders[split] = sorted([x.name for x in d.iterdir() if x.is_dir()]) if d.is_dir() else []
    ok = names == folders["train"] == folders["val"] and len(names) == 3
    if ok:
        print(f"  class contract : OK - {len(names)} classes, json equals sorted(folder names)")
        print(f"  index order    : {names}")
    else:
        print(f"  class contract : MISMATCH json={names} train={folders['train']} val={folders['val']}")
    return ok, names


def per_class_report(cm, names: list[str]) -> None:
    import numpy as np

    cm = np.asarray(cm, dtype=float)
    n = cm.shape[0]
    print("\n  per-class precision / recall / F1 (rows=true, cols=pred):")
    for i in range(n):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"    {names[i]:<12} P={p:.4f} R={r:.4f} F1={f1:.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Train the AgroVeyra triage classifier.")
    ap.add_argument("--data", type=str, default="data/triage")
    ap.add_argument("--model", type=str, default="yolo11n-cls.pt")
    ap.add_argument("--name", type=str, default="pest_triage")
    ap.add_argument("--project", type=str, default="runs")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=str, default="16")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--env-only", action="store_true")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--skip-contract-check", action="store_true")
    args = ap.parse_args()

    configure_stdout()

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (HERE / data_dir).resolve()
    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = (HERE / project_dir).resolve()

    cuda_ready, vram_gb = preflight()

    if args.env_only:
        return 0

    if not cuda_ready:
        print("\nNo CUDA device; refusing to train (CPU days are not worth it).")
        return 2

    data_ready = (data_dir / "train").is_dir() and (data_dir / "val").is_dir()
    if not data_ready:
        print(f"\ndataset not found: {data_dir} - build it first (ml/build_triage_dataset.py --apply)")
        return 2

    if args.skip_contract_check:
        print("\n  class contract : SKIPPED")
        names = ["disease", "healthy", "pest_damage"]
    else:
        ok, names = class_contract(data_dir)
        if not ok:
            print("\nrefusing to train: triage_class_names.json disagrees with folder order.")
            return 3

    suggested, note = suggest_batch(vram_gb)
    if args.batch.lower() == "auto":
        batch: int | str = -1
        print(f"\n  batch : -1 (auto; expect ~{suggested})")
    else:
        batch = int(args.batch)
        print(f"\n  batch : {batch}")

    print(f"  dataset : {data_dir}")
    print(f"  project : {project_dir}")
    print(f"  model   : {args.model}")
    print(f"  epochs  : {args.epochs}  imgsz={args.imgsz}  patience={args.patience}  seed={args.seed}")
    print("=" * 96)

    if args.check_only:
        print("\n--check-only: environment, dataset and contract verified. Not training.")
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
    print(f"\nTraining complete. Results: {save_dir}")

    trained_names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    print(f"  model.names matches triage_class_names.json: {trained_names == names}")
    if trained_names != names:
        print(f"  WARNING: model.names={trained_names} json={names}")

    try:
        val = model.val(data=str(data_dir), split="val", project=str(project_dir), name=args.name + "_val")
        print(f"  val top-1/top-5 : {val.top1:.4f} / {val.top5:.4f}")
        cm = getattr(getattr(val, "confusion_matrix", None), "matrix", None)
        if cm is not None:
            per_class_report(cm, names)
    except Exception as e:
        print(f"  val metrics unavailable: {e}")

    print(f"\nNext: export to TFLite and verify parity.")
    print(f"  checkpoint: {best}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
