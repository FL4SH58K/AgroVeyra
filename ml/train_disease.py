"""Train the AgroVeyra disease classifier (YOLO11n-cls) with an environment preflight.

The previous version of this file hardcoded `resume='runs/.../last.pt'` and `batch=32`, which meant:

  * it *continued an old run* instead of training a fresh model on a cleaned dataset, and
  * it silently assumed the 6 GB GPU that produced `disease_classifier-3`.

Training now starts fresh unless `--resume` is passed explicitly, and the batch size is chosen from
the machine it is actually running on. Before the first epoch it prints the interpreter, CUDA build,
GPU name, VRAM and ultralytics version, so moving to another laptop is visible in the log instead of
failing 40 minutes into epoch 1.

It also refuses to start unless `ml/verify_class_contract.py` passes: `backend/models/class_names.json`
(and its byte-identical Android copy) *is* the softmax index -> label mapping, so a dataset whose class
folders had drifted from that list would train happily and then predict the wrong crop with full
confidence. The 60-class set is the implementation target (the "52 categories" line in
GROUP_5_PROJECT.pdf is documentation only). `--skip-contract-check` bypasses the gate.

Usage:
    ml/.venv/Scripts/python.exe ml/train_disease.py --env-only
    ml/.venv/Scripts/python.exe ml/train_disease.py --data data/merged_clean --name disease_clean
    ml/.venv/Scripts/python.exe ml/train_disease.py --data data/merged_clean --batch 16
    ml/.venv/Scripts/python.exe ml/verify_class_contract.py --data ml/data/merged_clean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Conservative batch guidance by *total* VRAM for YOLO11n-cls at 224 px with AMP. Anything below
# 4 GB is effectively a CPU-class device for a 176 k image dataset, so say that out loud instead of
# silently swapping to a smaller batch and running for a week.
VRAM_BATCH_TABLE = (
    (4.0, 8, "consider --imgsz 160 as well"),
    (6.0, 16, "the RTX 3050 run used batch=32; 16 is the safe default at this size"),
    (8.0, 32, "matches the run that produced disease_classifier-3"),
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
        print("  ultralytics : NOT INSTALLED - pip install ultralytics before training")
        return False, 0.0

    print(f"  python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"  ultralytics : {ultralytics.__version__}")

    try:
        import torch
    except ImportError:
        print("  torch       : NOT INSTALLED - install torch before training")
        return False, 0.0

    print(f"  torch       : {torch.__version__}")
    print(f"  torch cuda  : {torch.version.cuda or 'not a CUDA build'}")

    if not torch.cuda.is_available():
        print("  cuda ready  : False")
        print("  => no usable CUDA device on this machine; train on a CUDA host or expect days on CPU.")
        return False, 0.0

    properties = torch.cuda.get_device_properties(0)
    vram_gb = properties.total_memory / 1024**3
    print("  cuda ready  : True")
    print(
        f"  device      : {properties.name} "
        f"({vram_gb:.1f} GB VRAM, capability {properties.major}.{properties.minor}, "
        f"{torch.cuda.device_count()} device(s) visible)"
    )
    return True, vram_gb


def suggest_batch(vram_gb: float) -> tuple[int, str]:
    for limit, batch, note in VRAM_BATCH_TABLE:
        if vram_gb < limit:
            return batch, note
    return 8, ""


def class_contract_preflight(data_dir: Path, expect_count: int, dataset_ready: bool) -> int:
    """Run ml/verify_class_contract.py in-process; 0 means the shipping class contract holds.

    The check is imported instead of re-implemented so the two can never drift. It asserts the contract
    JSON (60 names, Python sorted order), the Android copies of `class_names.json`,
    `treatment_db.json` and `agroveyra_model.tflite`, the flatbuffer fingerprint recorded in
    `ml/export_parity_report.txt`, and the `train/` + `val/` class folders of the dataset being used -
    which is what stops a dataset that disagrees with the app from being trained on at all.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from verify_class_contract import main as contract_main

    argv = ["--expect-count", str(expect_count)]
    argv += ["--data", str(data_dir)] if dataset_ready else ["--no-dataset"]
    return contract_main(argv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/merged_clean", help="classification dataset root (train/ + val/)")
    parser.add_argument("--model", default="yolo11n-cls.pt", help="starting weights when not resuming")
    parser.add_argument("--name", default="disease_clean", help="run name under --project")
    parser.add_argument("--project", default="runs", help="run output directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=224, help="keep 224: TFLiteHelper hardcodes INPUT_SIZE=224")
    parser.add_argument("--batch", default="auto", help="'auto' = Ultralytics auto-batch from live free VRAM, or an integer")
    parser.add_argument("--device", default="0", help="'0', '0,1' or 'cpu'")
    parser.add_argument("--workers", type=int, default=4, help="lower to 2 on Windows if the loader stalls")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr0", type=float, default=0.001)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", default=None, help="checkpoint to continue from (off by default)")
    parser.add_argument("--env-only", action="store_true", help="print the preflight and exit")
    parser.add_argument("--allow-missing-data", action="store_true", help="skip the dataset existence check")
    parser.add_argument("--skip-contract-check", action="store_true",
                        help="do not verify class_names.json against the dataset (unsafe)")
    parser.add_argument("--expect-count", type=int, default=60,
                        help="number of classes the contract must have (default 60)")
    return parser.parse_args()


def main() -> int:
    configure_stdout()
    args = parse_args()

    cuda_ready, vram_gb = preflight()
    suggested, note = suggest_batch(vram_gb)

    if args.env_only:
        print("")
        print(f"  suggested batch on this device: {suggested}{f' - {note}' if note else ''}")
        return 0

    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = (HERE / data_dir).resolve()
    # Ultralytics resolves a relative `project` against the *current directory*, so `--project runs`
    # launched from the repository root would write a stray `<root>/runs/disease_clean` and the
    # runbook's step 6 would then look for the checkpoint in ml/runs and not find it. Resolve it
    # against ml/ for the same reason --data is.
    project_dir = Path(args.project)
    if not project_dir.is_absolute():
        project_dir = (HERE / project_dir).resolve()
    data_ready = (data_dir / "train").is_dir() and (data_dir / "val").is_dir()
    if not args.allow_missing_data and not data_ready:
        print("")
        print(f"dataset not found: {data_dir}")
        print("  build the cleaned split first (ml/GPU_RUNBOOK.md, step 2), or pass --allow-missing-data.")
        return 2

    # Contract gate: class_names.json decides what softmax index 0..59 means, so a split whose folders
    # disagree with it would train fine and be labelled wrongly by the app with no visible symptom.
    if args.skip_contract_check:
        print("")
        print("  class contract : SKIPPED (--skip-contract-check) - index -> label mapping unverified")
    else:
        print("")
        if class_contract_preflight(data_dir, args.expect_count, data_ready) != 0:
            print("")
            print("  refusing to train: the class contract does not hold (violations listed above).")
            print("  fix the dataset or the class list, then re-run; --skip-contract-check overrides this.")
            return 3

    print("")
    if args.batch == "auto":
        # -1 lets Ultralytics size the batch from live free VRAM on *this* device; the table above is
        # only the expectation to compare the log against.
        batch: int | str = -1
        print(f"  batch       : -1 (auto from free VRAM; expect about {suggested}{f' - {note}' if note else ''})")
    else:
        try:
            batch = int(args.batch)
        except ValueError:
            print(f"invalid --batch {args.batch!r}: pass an integer or 'auto'")
            return 2
        print(f"  batch       : {batch} (explicit)")

    print(f"  dataset     : {data_dir}")
    print(f"  project     : {project_dir}")
    print(f"  model       : {args.model}")
    print(f"  resume      : {args.resume or 'no (fresh run)'}")
    print("=" * 96)
    print("")

    from ultralytics import YOLO

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
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
            lr0=args.lr0,
            dropout=args.dropout,
            seed=args.seed,
        )

    save_dir = Path(results.save_dir)
    best = save_dir / "weights" / "best.pt"
    print(f"Training complete. Results saved to: {save_dir}")
    print("")
    print("Next steps (ml/GPU_RUNBOOK.md step 6): export, then verify.")
    print(f"  ml\\.venv\\Scripts\\python.exe ml\\export_tflite_float32.py --checkpoint {best} --copy")
    print(
        f"  ml\\.venv\\Scripts\\python.exe ml\\verify_export_parity.py --checkpoint {best} "
        "--val-dir ml\\data\\merged_clean\\val"
    )
    print("Note: step 6 paths in the runbook use ml/runs/<name>; --project/--data resolve against ml/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
