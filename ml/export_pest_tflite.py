"""Export the pest classifier to float32 TFLite and verify it op-by-op (Task C).

Reuses the pipeline proven on the disease model (ml/export_tflite_float32.py): torch -> NCHW ONNX ->
fold integer shape arithmetic -> onnx2tf flatbuffer_direct -> float32 TFLite with an NHWC input.
That route exists because Ultralytics' own LiteRT export cannot run on Windows.

Pest-specific differences:
  * class names come from ml/data/pest/ip102_scoped/pest_class_names.json (41 slugs, sorted order)
  * parity is sampled from the scoped *test* split, because there is no real-world pest photo set
  * the artifact is agroveyra_pest_model.tflite, and agroveyra_model.tflite is never touched

Usage:
    ml\\.venv\\Scripts\\python.exe ml\\export_pest_tflite.py
    ml\\.venv\\Scripts\\python.exe ml\\export_pest_tflite.py --copy
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

import export_tflite_float32 as ex  # noqa: E402 - sys.path is set above

CHECKPOINT = HERE / "runs" / "pest_classifier_scoped" / "weights" / "best.pt"
CLASS_JSON = HERE / "data" / "pest" / "ip102_scoped" / "pest_class_names.json"
TEST_DIR = HERE / "data" / "pest" / "ip102_scoped" / "test"
BUILD_DIR = HERE / "export_build_pest"
ARTIFACT_NAME = "agroveyra_pest_model.tflite"
EXPORT_PYTHON = Path(r"C:\Python314\python.exe")
APP_ASSET = ROOT / "android" / "app" / "src" / "main" / "assets" / ARTIFACT_NAME
BACKEND_MODEL = ROOT / "backend" / "models" / ARTIFACT_NAME
EXPECTED_CLASSES = 41


def load_pest_names() -> list[str]:
    names = json.loads(CLASS_JSON.read_text(encoding="utf-8"))
    if len(names) != EXPECTED_CLASSES:
        raise SystemExit(f"expected {EXPECTED_CLASSES} pest classes, {CLASS_JSON.name} has {len(names)}")
    return names


def sample_test_images(count: int) -> list[Path]:
    """One deterministic test image per class (alphabetical), so parity runs are reproducible."""
    images = []
    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        first = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in ex.IMAGE_SUFFIXES)
        if first:
            images.append(first[0])
    if len(images) < count:
        print(f"      note: {len(images)} classes have test images; sampling all of them")
    return images[:count]


def verify_parity(checkpoint: Path, artifact: Path, class_names: list[str], imgsz: int, samples: int) -> dict:
    """Compare fp32 TFLite probabilities with the PyTorch checkpoint on identical pixels.

    Same method as the disease export (feed the identical preprocessed tensor to both graphs);
    the only difference is where the images come from, since pests have no field photo set.
    """
    import torch
    from ultralytics import YOLO

    import evaluate_real_world as ev

    images = sample_test_images(samples)
    if not images:
        return {"checked": 0}

    network = YOLO(str(checkpoint)).model.eval().cpu()
    interpreter_class = ev.import_interpreter()
    interpreter = interpreter_class(model_path=str(artifact), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    layout = "app" if [int(dim) for dim in input_detail["shape"]][-1] == 3 else "native"

    agreements = 0
    max_difference = 0.0
    rows = []
    for path in images:
        pixels = ev.preprocess_image(path, imgsz, False)
        tensor = ev.build_input_tensor(pixels, input_detail, layout)
        interpreter.set_tensor(input_detail["index"], tensor)
        interpreter.invoke()
        scores = ev.ensure_probabilities(
            ev.dequantize_output(interpreter.get_tensor(output_detail["index"]), output_detail)
        )

        torch_input = torch.from_numpy(tensor.astype(np.float32))
        torch_input = torch_input.permute(0, 3, 1, 2) if layout == "app" else torch_input
        with torch.no_grad():
            output = network(torch_input)
            while isinstance(output, (tuple, list)):
                output = output[0]
            reference = output.reshape(-1).cpu().numpy().astype(np.float32)

        difference = float(np.max(np.abs(scores - reference)))
        max_difference = max(max_difference, difference)
        tflite_top = int(np.argmax(scores))
        torch_top = int(np.argmax(reference))
        agreements += int(tflite_top == torch_top)
        rows.append(
            {
                "image": f"{path.parent.name}/{path.name}",
                "tflite": class_names[tflite_top],
                "torch": class_names[torch_top],
                "tflite_conf": float(scores[tflite_top]),
                "torch_conf": float(reference[torch_top]),
                "max_abs_prob_diff": difference,
            }
        )

    return {
        "checked": len(images),
        "layout_used": layout,
        "agreements": agreements,
        "max_abs_prob_diff": max_difference,
        "rows": rows,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Export the scoped pest classifier to float32 TFLite.")
    parser.add_argument("--checkpoint", default=str(CHECKPOINT))
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--samples", type=int, default=20, help="test images used for the parity cross-check")
    parser.add_argument("--skip-onnx", action="store_true", help="reuse an existing ONNX graph")
    parser.add_argument("--skip-convert", action="store_true", help="reuse an existing float32 artifact")
    parser.add_argument("--copy", action="store_true", help="install the artifact as the pest model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ex.configure_stdout()

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")
    if not ex.DIRECT_LIBS.is_dir():
        raise SystemExit(f"converter libraries not found: {ex.DIRECT_LIBS}")

    build_dir = BUILD_DIR
    build_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = build_dir / "pest_best.onnx"
    raw_output_dir = build_dir / "onnx2tf"
    artifact = build_dir / "agroveyra_pest_model_float32.tflite"

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text, flush=True)
        lines.append(text)

    emit("=" * 118)
    emit("AgroVeyra PEST classifier - float32 TFLite export and verification")
    emit("=" * 118)
    emit(f"checkpoint    : {checkpoint} ({ex.human(checkpoint.stat().st_size)})")
    emit(f"build dir     : {build_dir}")
    emit(f"converter     : onnx2tf flatbuffer_direct via {EXPORT_PYTHON}")
    emit("")

    pest_names = load_pest_names()
    checkpoint_names = ex.checkpoint_class_names(checkpoint)
    emit(f"classes       : checkpoint={len(checkpoint_names)} json={len(pest_names)}")
    identical = checkpoint_names == pest_names
    emit(f"class order   : {'identical to pest_class_names.json' if identical else 'MISMATCH'}")
    if not identical:
        for index, (a, b) in enumerate(zip(checkpoint_names, pest_names)):
            if a != b:
                emit(f"  first mismatch at index {index}: checkpoint={a!r} json={b!r}")
                break
        raise SystemExit("refusing to export: the class order does not match the json")
    emit("")

    if args.skip_onnx and onnx_path.is_file():
        emit(f"[1/5] reusing ONNX   : {onnx_path} ({ex.human(onnx_path.stat().st_size)})")
    else:
        emit(f"[1/5] exporting ONNX : opset={args.opset} imgsz={args.imgsz} (NCHW float32)")
        ex.export_onnx(checkpoint, onnx_path, args.imgsz, args.opset)
        emit(f"      wrote {onnx_path} ({ex.human(onnx_path.stat().st_size)})")

    if args.skip_convert and artifact.is_file():
        emit(f"[2/5] reusing artifact: {artifact} ({ex.human(artifact.stat().st_size)})")
    else:
        emit("[2/5] converting with onnx2tf (flatbuffer_direct) -> float32 TFLite")
        conversion_source = ex.freeze_reshape_shapes(ex.fold_shape_arithmetic(onnx_path))
        ex.run_onnx2tf_direct(EXPORT_PYTHON, conversion_source, raw_output_dir)
        produced = ex.find_float32_tflite(raw_output_dir)
        shutil.copy2(produced, artifact)
        emit(f"      produced {produced.name} -> {artifact.name} ({ex.human(artifact.stat().st_size)})")

    emit("")
    emit("[3/5] OP-LEVEL VERIFICATION (read from the flatbuffer, not assumed from export flags)")
    details = ex.inspect_model(artifact)
    emit(f"      input tensor   : {details['input']}")
    emit(f"      output tensor  : {details['output']}")
    emit(f"      ops in graph   : {details['op_count']}")
    emit(f"      contains softmax: {details['has_softmax']}")
    emit(f"      contains quantize/dequantize ops: {details['has_quantize_ops']}")
    checks = [
        (f"input layout is NHWC {details['input_shape']} (matches TFLiteHelper)",
         details["input_layout"] == "NHWC"),
        ("input dtype is float32 (no int8 requantization)", details["input_dtype"] == "float32"),
        ("output dtype is float32", details["output_dtype"] == "float32"),
        ("no quantize/dequantize ops in graph", not details["has_quantize_ops"]),
        (f"class count is {EXPECTED_CLASSES}", details["classes"] == EXPECTED_CLASSES),
        ("class order matches pest_class_names.json", checkpoint_names == pest_names),
    ]
    for label, ok in checks:
        emit(f"      [{'PASS' if ok else 'FAIL'}] {label}")
    emit("")

    emit(f"[4/5] NUMERICAL PARITY vs the checkpoint ({args.samples} test images, app preprocessing)")
    parity = verify_parity(checkpoint, artifact, pest_names, args.imgsz, args.samples)
    if parity.get("checked"):
        emit(f"      layout used for the TFLite feed: {parity['layout_used']}")
        emit(f"      top-1 agreement : {parity['agreements']}/{parity['checked']}")
        emit(f"      max |p_tflite - p_torch| : {parity['max_abs_prob_diff']:.6f}")
        for row in parity["rows"]:
            emit(
                f"      {row['image']:<44} | {row['tflite']:<30} | {row['torch']:<30} | "
                f"{row['tflite_conf'] * 100:8.2f}% | {row['torch_conf'] * 100:8.2f}%"
            )
    emit("")

    report = {
        "checkpoint": str(checkpoint),
        "artifact": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": ex.sha256(artifact),
        "class_names_identical": identical,
        "details": details,
        "parity": parity,
    }
    (build_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    all_checks_passed = all(ok for _, ok in checks)
    parity_ok = bool(parity.get("checked")) and parity["agreements"] == parity["checked"]
    if args.copy and not (all_checks_passed and parity_ok):
        emit("[5/5] NOT INSTALLING: op-level checks or parity did not fully pass")
        emit(f"      checks_passed={all_checks_passed} parity={parity.get('agreements')}"
             f"/{parity.get('checked')}")
        args.copy = False

    if args.copy:
        emit("[5/5] INSTALLING ARTIFACT")
        for label, destination in (("android asset", APP_ASSET), ("backend model", BACKEND_MODEL)):
            shutil.copy2(artifact, destination)
            emit(f"      {label:<14}: {destination} ({ex.human(destination.stat().st_size)})")
        emit("      note: the disease model agroveyra_model.tflite is untouched.")
    else:
        emit("[5/5] copy skipped (pass --copy to install the pest asset and backend model)")

    emit("")
    (build_dir / "export_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    emit(f"report: {build_dir / 'export_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
