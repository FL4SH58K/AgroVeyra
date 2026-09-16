#!/usr/bin/env python
"""Re-export a float32 (un-quantized) TFLite classifier from the trained Ultralytics checkpoint.

Why this exists
---------------
The deployed `agroveyra_model.tflite` was produced by Ultralytics' LiteRT exporter
(`utils/export/litert.py`) with `quantize=8` (static INT8, FP32 graph I/O preserved).
That file has two problems, both confirmed by `ml/evaluate_real_world.py`:

  1. layout: the graph declares a channels-first input `[1, 3, 224, 224]` while
     `TFLiteHelper.convertBitmapToInputBuffer()` writes interleaved NHWC pixels
     (app parity 0/16 real photos, 1.11% on the model's own val split);
  2. static INT8 quantization crushes the output: softmax is quantized to 1/256 steps
     (logits scale 0.0491) and the same 180 val images score 43.9-47.2% instead of the
     96.11% the FP32 checkpoint reaches.

Both are fixed by one FP32 export, provided the exported graph declares an NHWC input so
that `TFLiteHelper` does not have to change.

How the export is produced on Windows
------------------------------------
Ultralytics 8.4.x can only run its own TFLite/LiteRT export on Linux/macOS
(`export_litert` asserts that, and `litert-torch` needs `jax` + `litert-converter`, which
have no Windows wheels). The Windows-supported equivalent is the classic
`PyTorch -> ONNX -> onnx2tf -> TFLite` path:

    ml/.venv           : torch + ultralytics  -> writes best.onnx (NCHW, opset 13)
    <export python>    : onnx2tf -> TFLite    -> onnx2tf transposes NCHW to NHWC

onnx2tf's default behaviour is exactly what the app needs: it inserts the transposes and
declares the TFLite input as NHWC `[1, 224, 224, 3]`, so no Kotlin/backend change is
required. This script verifies that claim at op level instead of assuming it.

The converter stack is installed into `ml/.export-libs` with
(`--target`) so that the working `ml/.venv` (numpy 2.4 / protobuf 7 / ai-edge-litert 2.2)
is never downgraded by TensorFlow's pins:

    C:\\Python311\\python.exe -m pip install --target ml/.export-libs \\
        tensorflow==2.19.0 "tf_keras<=2.19.0" "onnx2tf>=1.26.3,<1.29.0" \\
        "sng4onnx>=1.0.1" "onnx_graphsurgeon>=0.3.26" "onnxslim>=0.1.82" onnxruntime "onnx>=1.12,<2"

Usage (from the repository root):

    ml/.venv/Scripts/python.exe ml/export_tflite_float32.py           # convert + verify
    ml/.venv/Scripts/python.exe ml/export_tflite_float32.py --copy    # + overwrite app/backend
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent

DEFAULT_CHECKPOINT = (
    HERE / "runs" / "classify" / "runs" / "disease_classifier-3" / "weights" / "best.pt"
)
DEFAULT_BUILD_DIR = HERE / "export_build"
EXPORT_LIBS = HERE / ".export-libs"
DEFAULT_EXPORT_PYTHON = Path(r"C:\Python311\python.exe")
# onnx2tf >= 2.5 defaults to its TensorFlow-free "flatbuffer_direct" backend and needs Python >= 3.12.
DIRECT_LIBS = HERE / ".export-libs-314"
DEFAULT_DIRECT_PYTHON = Path(r"C:\Python314\python.exe")

APP_ASSET = PROJECT_ROOT / "android" / "app" / "src" / "main" / "assets" / "agroveyra_model.tflite"
BACKEND_MODEL = PROJECT_ROOT / "backend" / "models" / "agroveyra_model.tflite"
CLASS_NAMES_JSON = PROJECT_ROOT / "backend" / "models" / "class_names.json"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def configure_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def human(size: int) -> str:
    if size >= 1 << 20:
        return f"{size / (1 << 20):.2f} MiB"
    return f"{size / 1024:.1f} KiB"


def export_onnx(checkpoint: Path, onnx_path: Path, imgsz: int, opset: int) -> list[str]:
    """Load the checkpoint and write an NCHW ONNX graph; returns the class names.

    The Ultralytics classification network returns a ``(probabilities, logits)`` tuple in
    inference mode, which makes torch export two graph outputs. The wrapper below collapses
    that to exactly one tensor so the resulting TFLite has a single output and its semantics
    do not depend on output ordering.

    `torch` is imported here (not at module level) so that stub/flag handling stays cheap.
    """
    import torch
    from ultralytics import YOLO

    class ProbabilityHead(torch.nn.Module):
        """One output: class probabilities."""

        def __init__(self, network: torch.nn.Module, apply_softmax: bool) -> None:
            super().__init__()
            self.network = network
            self.apply_softmax = apply_softmax

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            output = self.network(inputs)
            while isinstance(output, (tuple, list)):
                output = output[0]
            output = output.reshape(output.shape[0], -1)
            return output.softmax(dim=1) if self.apply_softmax else output

    model = YOLO(str(checkpoint))
    if getattr(model, "task", None) != "classify":
        raise SystemExit(f"Checkpoint is a '{model.task}' model, expected 'classify'")

    names = [model.names[index] for index in sorted(model.names)]
    network = model.model.eval().cpu()
    dummy = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)

    with torch.no_grad():
        probe = network(dummy)
        while isinstance(probe, (tuple, list)):
            probe = probe[0]
        probe = probe.reshape(probe.shape[0], -1)
        already_probabilities = bool(probe.min() >= 0.0 and abs(float(probe.sum()) - 1.0) < 1e-3)
        print(
            f"      network raw output: shape={tuple(probe.shape)} sum={float(probe.sum()):.4f} "
            f"min={float(probe.min()):.6f} -> extra softmax: {not already_probabilities}"
        )
        wrapped = ProbabilityHead(network, apply_softmax=not already_probabilities)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapped,
            dummy,
            str(onnx_path),
            opset_version=opset,
            input_names=["images"],
            output_names=["output0"],
            do_constant_folding=True,
            dynamic_axes=None,
        )
    return names


SAMPLE_DATA_NAME = "calibration_image_sample_data_20x128x128x3_float32.npy"


def ensure_onnx2tf_sample_data(directory: Path, count: int = 20, size: int = 128) -> Path:
    """Make sure onnx2tf's reference sample data exists in a form numpy can load.

    onnx2tf reads ``<cwd>/<SAMPLE_DATA_NAME>`` with ``np.load`` (``allow_pickle=False``) and
    downloads it only when the file is missing. The copy published by onnx2tf is pickle
    encoded, which current numpy refuses to load ("Cannot load file containing pickled data"),
    so the official Ultralytics copy of the same sample data is fetched instead - the same
    work-around Ultralytics applies before calling onnx2tf. If the download is unavailable a
    deterministic float32 array is written so the conversion can proceed offline.
    """
    target = directory / SAMPLE_DATA_NAME
    if target.is_file():
        try:
            if np.load(target).shape[0] == count:
                return target
        except Exception:
            pass  # unreadable (pickled) copy: replace it below

    try:
        from ultralytics.utils.downloads import attempt_download_asset

        with contextlib.chdir(directory):  # attempt_download_asset writes into the cwd
            attempt_download_asset(f"{SAMPLE_DATA_NAME}.zip", unzip=True, delete=True)
        if target.is_file() and np.load(target).shape[0] == count:
            print(f"      sample data downloaded: {target.name}")
            return target
    except Exception as error:  # noqa: BLE001 - fall back to generated data
        print(f"      sample data download unavailable ({type(error).__name__}: {error})")

    np.save(target, np.random.default_rng(0).random((count, size, size, 3), dtype=np.float32))
    print(f"      sample data generated: {target.name}")
    return target


# Integer-only ops that torch.onnx uses for shape arithmetic (Shape -> Gather -> Div -> Mul -> ...).
_SHAPE_ARITHMETIC_OPS = {
    "Add",
    "Cast",
    "Ceil",
    "Concat",
    "Div",
    "Floor",
    "Gather",
    "Mul",
    "Reshape",
    "Slice",
    "Squeeze",
    "Sub",
    "Unsqueeze",
}


def _evaluate_shape_node(node, arrays):
    """Evaluate one integer shape-arithmetic ONNX node; returns None when unsupported."""
    attributes = {attribute.name: attribute for attribute in node.attribute}
    op = node.op_type
    if op == "Gather":
        axis = attributes["axis"].i if "axis" in attributes else 0
        return np.take(arrays[0], arrays[1].astype(np.int64), axis=axis)
    if op == "Slice":
        starts, ends = arrays[0].astype(np.int64), arrays[1].astype(np.int64)
        axes = arrays[2].astype(np.int64) if len(arrays) > 2 else np.arange(len(starts))
        steps = arrays[3].astype(np.int64) if len(arrays) > 3 else np.ones_like(starts)
        slices = [slice(None)] * arrays[0].ndim
        for start, end, axis, step in zip(starts, ends, axes, steps):
            slices[int(axis)] = slice(int(start), int(end), int(step))
        return arrays[0][tuple(slices)]
    if op == "Div":
        return np.trunc(arrays[0] / arrays[1]).astype(np.int64)
    if op == "Mul":
        return (arrays[0] * arrays[1]).astype(np.int64)
    if op == "Add":
        return (arrays[0] + arrays[1]).astype(np.int64)
    if op == "Sub":
        return (arrays[0] - arrays[1]).astype(np.int64)
    if op == "Concat":
        axis = attributes["axis"].i if "axis" in attributes else 0
        return np.concatenate(arrays, axis=axis)
    if op == "Cast":
        return arrays[0].astype(np.int64)  # only reached for integer inputs
    if op in {"Unsqueeze", "Squeeze"}:
        axes = [axis.i for axis in attributes["axes"].ints] if "axes" in attributes else None
        if op == "Unsqueeze":
            return np.expand_dims(arrays[0], tuple(axes)) if axes is not None else np.expand_dims(arrays[0], 0)
        return np.squeeze(arrays[0], tuple(axes) if axes is not None else None)
    if op == "Reshape":
        return arrays[0].reshape([int(value) for value in np.asarray(arrays[1]).reshape(-1)])
    if op == "Floor":
        return np.floor(arrays[0]).astype(np.int64)
    if op == "Ceil":
        return np.ceil(arrays[0]).astype(np.int64)
    return None


def fold_shape_arithmetic(onnx_path: Path) -> Path:
    """Freeze integer shape arithmetic (Shape/Gather/Div/Mul/...) into constants.

    ``torch.onnx.export`` expresses size-dependent logic (``chunk``/``split``/``reshape``) as integer
    tensor arithmetic. onnx2tf wires those symbolic tensors into ``Slice``/``strided_slice``
    incorrectly, which yields zero-sized tensors and aborts the TensorFlow build with
    "Number of groups must not be 0". The graph input shape is fixed here, so the arithmetic is
    evaluated and written back as initializers. Float tensors are never touched: a node is folded
    only when every input is an integer constant.
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(model)

    shape_of: dict[str, np.ndarray] = {}
    for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]:
        tensor_type = value.type.tensor_type
        if tensor_type.HasField("shape"):
            shape_of[value.name] = np.asarray(
                [int(dim.dim_value) if dim.HasField("dim_value") else -1 for dim in tensor_type.shape.dim],
                dtype=np.int64,
            )

    known: dict[str, np.ndarray] = {
        initializer.name: numpy_helper.to_array(initializer) for initializer in model.graph.initializer
    }
    for node in model.graph.node:  # constants expressed as nodes rather than initializers
        if node.op_type == "Constant":
            tensor = next((attribute.t for attribute in node.attribute if attribute.name == "value"), None)
            if tensor is not None:
                known[node.output[0]] = numpy_helper.to_array(tensor)

    frozen: dict[str, np.ndarray] = {}
    removable: list = []
    for node in model.graph.node:
        if node.op_type == "Shape" and node.input and node.input[0] in shape_of:
            frozen[node.output[0]] = shape_of[node.input[0]]
            removable.append(node)
            continue
        if node.op_type not in _SHAPE_ARITHMETIC_OPS:
            continue
        arrays = []
        for name in node.input:
            array = frozen.get(name, known.get(name))
            if array is None:
                break
            arrays.append(array)
        else:
            if arrays and all(array.dtype.kind in "iu" for array in arrays):  # never touch float data ops
                try:
                    result = _evaluate_shape_node(node, arrays)
                except Exception:
                    result = None
                if result is not None:
                    frozen[node.output[0]] = np.asarray(result, dtype=np.int64)
                    removable.append(node)

    if not frozen:
        print("      no integer shape arithmetic found (nothing to fold)")
        return onnx_path

    renamed: dict[str, str] = {}
    for name, array in frozen.items():
        new_name = "frozen__" + name.replace("/", "_").replace(":", "_")
        model.graph.initializer.append(numpy_helper.from_array(array, name=new_name))
        renamed[name] = new_name

    removed = {id(node) for node in removable}
    kept = []
    for node in model.graph.node:
        if id(node) in removed:
            continue
        for index, name in enumerate(node.input):
            if name in renamed:
                node.input[index] = renamed[name]
        kept.append(node)
    del model.graph.node[:]
    model.graph.node.extend(kept)

    folded_path = onnx_path.with_name(f"{onnx_path.stem}_folded.onnx")
    onnx.save(model, str(folded_path))
    print(f"      froze {len(frozen)} integer shape-arithmetic tensor(s) -> {folded_path.name}")
    return folded_path


def freeze_reshape_shapes(onnx_path: Path) -> Path:
    """Fold the symbolic batch dimension out of the graph.

    ``torch.onnx.export`` writes ``x.view(x.size(0), -1)`` as ``Shape -> Gather -> Concat -> Reshape``,
    which leaves the batch dimension symbolic ('unk__75' in onnx2tf's log). TensorFlow's convolution
    shape inference then collapses that dimension to 0 and onnx2tf aborts with
    "Number of groups must not be 0". The graph input is fixed at ``[1, 3, H, W]``, so the inferred
    static shapes are written back as Reshape constants (shape inference is run over the whole graph,
    so every Reshape that only depends on the input shape becomes static).
    """
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(onnx_path))
    inferred = onnx.shape_inference.infer_shapes(model)
    value_shapes = {
        value.name: [
            int(dim.dim_value) if dim.HasField("dim_value") else None
            for dim in value.type.tensor_type.shape.dim
        ]
        for value in [*inferred.graph.input, *inferred.graph.value_info, *inferred.graph.output]
    }
    initializers = {init.name for init in model.graph.initializer}
    folded: dict[str, str] = {}
    changed = 0
    for node in model.graph.node:
        if node.op_type != "Reshape" or len(node.input) < 2:
            continue
        shape_name = node.input[1]
        if shape_name in initializers:
            continue
        target = value_shapes.get(node.output[0])
        if not target or any(dim is None or dim < 0 for dim in target):
            continue
        if shape_name not in folded:
            constant_name = f"{shape_name}__static"
            model.graph.initializer.append(
                numpy_helper.from_array(np.asarray(target, dtype=np.int64), name=constant_name)
            )
            folded[shape_name] = constant_name
            initializers.add(constant_name)
        node.input[1] = folded[shape_name]
        changed += 1

    if not changed:
        print("      no symbolic Reshape shape inputs found (nothing to fold)")
        return onnx_path

    static_path = onnx_path.with_name(f"{onnx_path.stem}_static.onnx")
    onnx.save(model, str(static_path))
    print(f"      folded {changed} symbolic Reshape shape input(s) -> {static_path.name}")
    return static_path


def run_onnx2tf_direct(export_python: Path, onnx_path: Path, output_dir: Path) -> None:
    """Convert with onnx2tf's TensorFlow-free ``flatbuffer_direct`` backend.

    This is the route that works on this machine: Ultralytics 8.4 can only run its own LiteRT export
    on Linux/macOS (``litert-torch`` needs ``jax``/``litert-converter``, which have no Windows wheels),
    and the TensorFlow-based converter aborts on this architecture's channel-split ``Slice``
    ("Number of groups must not be 0" / mismatched ``Add`` operands). onnx2tf >= 2.5 replaced that
    backend with a ModelIR based one that handles the graph and emits a float32 NHWC model.
    """
    if not export_python.is_file():
        raise SystemExit(f"export python not found: {export_python}")
    if not DIRECT_LIBS.is_dir():
        raise SystemExit(
            f"converter libraries not found: {DIRECT_LIBS}\n"
            "install them with:\n"
            f'  {export_python} -m pip install --target {DIRECT_LIBS} numpy onnx flatbuffers ml-dtypes ai-edge-litert onnxruntime\n'
            f'  {export_python} -m pip install --target {DIRECT_LIBS} --no-deps "onnx2tf>=2.6"\n'
            f'  {export_python} -m pip install --target {DIRECT_LIBS} psutil setuptools sng4onnx sne4onnx'
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{DIRECT_LIBS}{os.pathsep}{existing}" if existing else str(DIRECT_LIBS)
    environment["PYTHONIOENCODING"] = "utf-8"

    command = [
        str(export_python),
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(output_dir),
        "-tb",
        "flatbuffer_direct",
    ]
    print("[export] running:", " ".join(command))
    completed = subprocess.run(
        command,
        env=environment,
        cwd=str(output_dir),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (output_dir / "onnx2tf.log").write_text(
        (completed.stdout or "") + "\n--- stderr ---\n" + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0:
        tail = (completed.stdout or "")[-4000:] + (completed.stderr or "")[-4000:]
        raise SystemExit(f"onnx2tf (flatbuffer_direct) failed (rc={completed.returncode}):\n{tail}")


def run_onnx2tf(export_python: Path, onnx_path: Path, output_dir: Path, keep_nchw: bool = False) -> None:
    """Invoke onnx2tf in the detached converter environment (TensorFlow + onnx2tf)."""
    if not export_python.is_file():
        raise SystemExit(f"export python not found: {export_python}")
    if not EXPORT_LIBS.is_dir():
        raise SystemExit(
            f"converter libraries not found: {EXPORT_LIBS}\n"
            "install them with:\n"
            f"  {export_python} -m pip install --target {EXPORT_LIBS} "
            "tensorflow==2.19.0 tf_keras onnx2tf onnxslim onnxruntime onnx"
        )
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_onnx2tf_sample_data(output_dir)

    environment = dict(os.environ)
    existing = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = f"{EXPORT_LIBS}{os.pathsep}{existing}" if existing else str(EXPORT_LIBS)
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["TF_CPP_MIN_LOG_LEVEL"] = "2"

    command = [
        str(export_python),
        "-m",
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(output_dir),
        "-osd",  # write the TF SavedModel signature definitions needed by the converter
        "-nuo",  # skip onnxsim: it needs a console script that --target installs do not provide
    ]
    if keep_nchw:
        # Ask onnx2tf not to transpose the channels-first input to NHWC. Its transpose rewrite
        # breaks on this architecture's squeeze-excite shape chain, so the channels-first graph is
        # converted as-is (the app then has to feed CHW planes, see TFLiteHelper).
        command += ["-k", "images"]
    print("[export] running:", " ".join(command))
    completed = subprocess.run(
        command,
        env=environment,
        cwd=str(output_dir),  # onnx2tf resolves its reference sample data from the cwd
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    (output_dir / "onnx2tf.log").write_text(
        (completed.stdout or "") + "\n--- stderr ---\n" + (completed.stderr or ""), encoding="utf-8"
    )
    if completed.returncode != 0:
        tail = (completed.stdout or "")[-4000:] + (completed.stderr or "")[-4000:]
        raise SystemExit(f"onnx2tf failed (rc={completed.returncode}):\n{tail}")


def find_float32_tflite(output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*float32*.tflite"))
    if not candidates:
        all_models = sorted(output_dir.rglob("*.tflite"))
        raise SystemExit(
            "no *_float32.tflite produced. tflite files present: "
            + ", ".join(str(path) for path in all_models)
        )
    return candidates[0]


def describe(detail: dict) -> str:
    shape = [int(dim) for dim in detail["shape"]]
    scale, zero_point = (float(v) for v in detail["quantization"])
    quant = "none" if scale == 0 else f"scale={scale:g} zero_point={zero_point:g}"
    return f"shape={shape} dtype={np.dtype(detail['dtype']).name} quantization={quant}"


def inspect_model(model_path: Path) -> dict:
    """Op-level inspection of the exported model (never trust the export flags)."""
    sys.path.insert(0, str(HERE))
    import evaluate_real_world as ev

    interpreter_class = ev.import_interpreter()
    interpreter = interpreter_class(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]

    op_names: list[str] = []
    if hasattr(interpreter, "_get_ops_details"):
        try:
            op_names = [str(op["op_name"]) for op in interpreter._get_ops_details()]
        except Exception:
            op_names = []

    shape = [int(dim) for dim in input_detail["shape"]]
    return {
        "input": describe(input_detail),
        "output": describe(output_detail),
        "input_shape": shape,
        "input_dtype": np.dtype(input_detail["dtype"]).name,
        "output_dtype": np.dtype(output_detail["dtype"]).name,
        "input_layout": "NHWC" if shape[-1] == 3 else "NCHW" if len(shape) == 4 and shape[1] == 3 else "unknown",
        "classes": int(np.prod(output_detail["shape"])),
        "op_count": len(op_names),
        "has_quantize_ops": any("QUANTIZE" in name.upper() or "DEQUANTIZE" in name.upper() for name in op_names),
        "has_softmax": any("SOFTMAX" in name.upper() for name in op_names),
    }


def verify_against_checkpoint(
    checkpoint: Path, model_path: Path, class_names: list[str], samples: int, imgsz: int
) -> dict:
    """Compare fp32 TFLite probabilities with the PyTorch checkpoint on identical pixels.

    The same preprocessed tensor is fed to both graphs (``model.predict`` would apply the
    Ultralytics validation transform instead, which compares different pixels and makes the
    difference meaningless on borderline images).
    """
    import torch
    from ultralytics import YOLO

    sys.path.insert(0, str(HERE))
    import evaluate_real_world as ev

    images_dir = HERE / "real_world_test"
    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)[:samples]
    if not images:
        return {"checked": 0}

    network = YOLO(str(checkpoint)).model.eval().cpu()
    interpreter_class = ev.import_interpreter()
    interpreter = interpreter_class(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]
    output_detail = interpreter.get_output_details()[0]
    layout = "app" if [int(d) for d in input_detail["shape"]][-1] == 3 else "native"

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
                "image": path.name,
                "tflite": class_names[tflite_top],
                "torch": class_names[torch_top],
                "tflite_conf": float(scores[tflite_top]),
                "torch_conf": float(reference[torch_top]),
                "max_abs_prob_diff": difference,
            }
        )
    return {
        "checked": len(images),
        "agreements": agreements,
        "layout_used": layout,
        "max_abs_prob_diff": max_difference,
        "rows": rows,
    }


def checkpoint_class_names(checkpoint: Path) -> list[str]:
    from ultralytics import YOLO

    model = YOLO(str(checkpoint))
    if not model.names:
        raise SystemExit(f"checkpoint {checkpoint} carries no class names")
    return [model.names[index] for index in sorted(model.names)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="Ultralytics checkpoint to export")
    parser.add_argument("--build-dir", default=str(DEFAULT_BUILD_DIR), help="scratch folder for ONNX/export output")
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument(
        "--converter",
        choices=("auto", "direct", "tensorflow"),
        default="auto",
        help="auto = flatbuffer_direct when ml/.export-libs-314 exists, else the TensorFlow backend",
    )
    parser.add_argument("--export-python", default=None, help="interpreter owning the converter libraries")
    parser.add_argument("--skip-onnx", action="store_true", help="reuse an existing best.onnx in the build dir")
    parser.add_argument("--onnx-only", action="store_true", help="stop after writing the ONNX graph")
    parser.add_argument("--skip-convert", action="store_true", help="reuse an existing float32 artifact")
    parser.add_argument(
        "--keep-nchw",
        action="store_true",
        help="ask onnx2tf to keep the channels-first input (no NHWC transpose rewrite)",
    )
    parser.add_argument("--samples", type=int, default=8, help="real-world images used for the checkpoint cross-check")
    parser.add_argument("--copy", action="store_true", help="overwrite the Android asset and the backend model")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configure_stdout()

    sys.path.insert(0, str(HERE))
    import evaluate_real_world as ev

    checkpoint = Path(args.checkpoint).resolve()
    build_dir = Path(args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = build_dir / "best.onnx"
    raw_output_dir = build_dir / "onnx2tf"
    artifact = build_dir / "agroveyra_model_float32.tflite"

    converter = args.converter
    if converter == "auto":
        converter = "direct" if DIRECT_LIBS.is_dir() else "tensorflow"
    export_python = (
        Path(args.export_python)
        if args.export_python
        else (DEFAULT_DIRECT_PYTHON if converter == "direct" else DEFAULT_EXPORT_PYTHON)
    )

    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit("=" * 118)
    emit("AgroVeyra float32 (un-quantized) TFLite export")
    emit("=" * 118)
    emit(f"checkpoint    : {checkpoint} ({human(checkpoint.stat().st_size)})")
    emit(f"build dir     : {build_dir}")
    emit(f"converter     : onnx2tf {converter} via {export_python}")

    checkpoint_names = checkpoint_class_names(checkpoint)
    service_names = ev.load_class_names(CLASS_NAMES_JSON)
    emit(f"classes       : checkpoint={len(checkpoint_names)} class_names.json={len(service_names)}")
    emit(f"class order   : {'identical to class_names.json' if checkpoint_names == service_names else 'MISMATCH'}")
    emit("")

    if args.skip_onnx and onnx_path.is_file():
        emit(f"[1/5] reusing ONNX    : {onnx_path} ({human(onnx_path.stat().st_size)})")
    else:
        emit(f"[1/5] exporting ONNX  : opset={args.opset} imgsz={args.imgsz} (NCHW float32)")
        export_onnx(checkpoint, onnx_path, args.imgsz, args.opset)
        emit(f"      wrote {onnx_path} ({human(onnx_path.stat().st_size)})")

    if args.onnx_only:
        emit("[2/5] --onnx-only: stopping before the TFLite conversion")
        return 0

    if args.skip_convert and artifact.is_file():
        emit(f"[2/5] reusing artifact: {artifact} ({human(artifact.stat().st_size)})")
    else:
        emit(f"[2/5] converting with onnx2tf ({converter}) -> TFLite, float32")
        conversion_source = freeze_reshape_shapes(fold_shape_arithmetic(onnx_path))
        if converter == "direct":
            run_onnx2tf_direct(export_python, conversion_source, raw_output_dir)
        else:
            run_onnx2tf(export_python, conversion_source, raw_output_dir, keep_nchw=args.keep_nchw)
        produced = find_float32_tflite(raw_output_dir)
        shutil.copy2(produced, artifact)
        emit(f"      produced {produced.name} -> {artifact} ({human(artifact.stat().st_size)})")

    emit("")
    emit("[3/5] OP-LEVEL VERIFICATION (read from the flatbuffer, not assumed from export flags)")
    details = inspect_model(artifact)
    emit(f"      input tensor   : {details['input']}")
    emit(f"      output tensor  : {details['output']}")
    emit(f"      ops in graph   : {details['op_count']}")
    emit(f"      contains softmax: {details['has_softmax']}")
    emit(f"      contains quantize/dequantize ops: {details['has_quantize_ops']}")
    checks = [
        (
            f"input layout is NHWC {details['input_shape']} (matches TFLiteHelper)",
            details["input_layout"] == "NHWC",
        ),
        ("input dtype is float32 (no int8 requantization)", details["input_dtype"] == "float32"),
        ("output dtype is float32", details["output_dtype"] == "float32"),
        ("no quantize/dequantize ops in graph", not details["has_quantize_ops"]),
        ("class count matches class_names.json", details["classes"] == len(service_names)),
        ("class order matches class_names.json", checkpoint_names == service_names),
    ]
    for label, ok in checks:
        emit(f"      [{'PASS' if ok else 'FAIL'}] {label}")
    emit("")

    emit(f"[4/5] NUMERICAL PARITY vs the checkpoint ({args.samples} real-world photos, app preprocessing)")
    parity = verify_against_checkpoint(checkpoint, artifact, service_names, args.samples, args.imgsz)
    if parity.get("checked"):
        emit(f"      layout used for the TFLite feed: {parity['layout_used']}")
        emit(f"      top-1 agreement : {parity['agreements']}/{parity['checked']}")
        emit(f"      max |p_tflite - p_torch| : {parity['max_abs_prob_diff']:.6f}")
        emit(f"      {'image':<34} | {'tflite':<32} | {'torch':<32} | tflite conf | torch conf | max dp")
        for row in parity["rows"]:
            emit(
                f"      {row['image']:<34} | {row['tflite']:<32} | {row['torch']:<32} | "
                f"{row['tflite_conf'] * 100:10.2f}% | {row['torch_conf'] * 100:9.2f}% | {row['max_abs_prob_diff']:.6f}"
            )
    emit("")

    report = {
        "checkpoint": str(checkpoint),
        "artifact": str(artifact),
        "artifact_size_bytes": artifact.stat().st_size,
        "artifact_sha256": sha256(artifact),
        "class_names_identical": checkpoint_names == service_names,
        "details": details,
        "parity": parity,
    }
    (build_dir / "export_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.copy:
        emit("[5/5] INSTALLING ARTIFACT")
        for label, destination in (("android asset", APP_ASSET), ("backend model", BACKEND_MODEL)):
            shutil.copy2(artifact, destination)
            emit(
                f"      {label:<14}: {destination} ({human(destination.stat().st_size)}, "
                f"sha256 {sha256(destination)[:16]})"
            )
        emit(f"      source artifact: {artifact}")
        emit("      note: android/app/build/intermediates/assets/debug/agroveyra_model.tflite is a")
        emit("            previously built copy; the next Gradle build regenerates it.")
    else:
        emit("[5/5] copy skipped (pass --copy to overwrite the Android asset and backend model)")

    emit("")
    emit(f"report: {build_dir / 'export_report.json'}")
    (build_dir / "export_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
