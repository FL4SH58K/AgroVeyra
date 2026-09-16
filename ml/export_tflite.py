"""Convert the trained AgroVeyra model to TFLite and verify the export.

This script:
1. Loads models/final_model.keras.
2. Applies DEFAULT optimization and full integer quantization using a
   representative dataset drawn from 100 validation images.
3. Saves the converted model to models/agroveyra_model.tflite.
4. Runs a single inference through the TFLite interpreter to verify it works.
5. Prints model size and inference time.
6. Ensures models/class_names.json exists by copying it if needed.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ML_DIR = PROJECT_ROOT / "ml"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_VALID_DIR = PROJECT_ROOT / "data" / "plant_disease" / "valid"
SAVED_MODEL_PATH = MODELS_DIR / "final_model.keras"
TFLITE_MODEL_PATH = MODELS_DIR / "agroveyra_model.tflite"
CLASS_NAMES_SOURCE_CANDIDATES = [
	MODELS_DIR / "class_names.json",
	PROJECT_ROOT / "class_names.json",
	ML_DIR / "class_names.json",
]
IMAGE_SIZE = (224, 224)


def print_error(message: str) -> None:
	"""Print a consistent error message to stderr."""

	print(f"[ERROR] {message}", file=sys.stderr)


def ensure_class_names_file() -> Path:
	"""Copy class_names.json into models/ if it is available elsewhere."""

	MODELS_DIR.mkdir(parents=True, exist_ok=True)

	destination = MODELS_DIR / "class_names.json"
	if destination.exists():
		return destination

	for source in CLASS_NAMES_SOURCE_CANDIDATES:
		if source.exists():
			shutil.copy2(source, destination)
			return destination

	raise FileNotFoundError(
		"class_names.json was not found in models/, the project root, or ml/. "
		"Run the training script first or place class_names.json in one of those locations."
	)


def find_validation_images() -> list[Path]:
	"""Return validation images for representative dataset calibration."""

	if not DATASET_VALID_DIR.exists():
		raise FileNotFoundError(
			f"Validation dataset directory not found: {DATASET_VALID_DIR}. "
			"The exporter needs validation images for representative calibration."
		)

	image_paths = sorted(
		path for path in DATASET_VALID_DIR.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
	)

	if len(image_paths) < 100:
		raise ValueError(
			f"Expected at least 100 validation images for calibration, found {len(image_paths)}."
		)

	return image_paths[:100]


def load_and_prepare_image(image_path: Path) -> np.ndarray:
	"""Load a single image and prepare it for the saved Keras model."""

	image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
	image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
	image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
	return np.expand_dims(image_array, axis=0)


def representative_dataset_generator() -> object:
	"""Yield calibration samples for full integer quantization."""

	for image_path in find_validation_images():
		image = tf.keras.utils.load_img(image_path, target_size=IMAGE_SIZE)
		image_array = tf.keras.utils.img_to_array(image).astype(np.float32)
		image_array = tf.keras.applications.efficientnet.preprocess_input(image_array)
		yield [np.expand_dims(image_array, axis=0)]


def convert_to_tflite() -> bytes:
	"""Convert the saved Keras model to a fully quantized TFLite model."""

	if not SAVED_MODEL_PATH.exists():
		raise FileNotFoundError(
			f"Saved model not found: {SAVED_MODEL_PATH}. Train the model before exporting."
		)

	model = tf.keras.models.load_model(SAVED_MODEL_PATH)

	converter = tf.lite.TFLiteConverter.from_keras_model(model)
	converter.optimizations = [tf.lite.Optimize.DEFAULT]
	converter.representative_dataset = representative_dataset_generator
	converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
	converter.inference_input_type = tf.int8
	converter.inference_output_type = tf.int8

	return converter.convert()


def save_tflite_model(tflite_model: bytes) -> None:
	"""Persist the TFLite model to disk."""

	MODELS_DIR.mkdir(parents=True, exist_ok=True)
	TFLITE_MODEL_PATH.write_bytes(tflite_model)


def run_tflite_verification() -> tuple[float, np.ndarray]:
	"""Load the TFLite model and run a single inference for verification."""

	validation_images = find_validation_images()
	sample_image_path = validation_images[0]
	sample_input = load_and_prepare_image(sample_image_path)

	interpreter = tf.lite.Interpreter(model_path=str(TFLITE_MODEL_PATH))
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	input_index = input_details[0]["index"]
	output_index = output_details[0]["index"]
	input_dtype = input_details[0]["dtype"]

	if input_dtype == np.int8:
		input_scale, input_zero_point = input_details[0]["quantization"]
		if input_scale == 0:
			raise ValueError("Invalid input quantization scale reported by the TFLite interpreter.")
		sample_input = np.round(sample_input / input_scale + input_zero_point).astype(np.int8)
	elif input_dtype != np.float32:
		raise TypeError(f"Unsupported TFLite input dtype: {input_dtype}")

	start_time = time.perf_counter()
	interpreter.set_tensor(input_index, sample_input)
	interpreter.invoke()
	inference_ms = (time.perf_counter() - start_time) * 1000.0

	output_data = interpreter.get_tensor(output_index)
	if output_details[0]["dtype"] == np.int8:
		output_scale, output_zero_point = output_details[0]["quantization"]
		if output_scale == 0:
			raise ValueError("Invalid output quantization scale reported by the TFLite interpreter.")
		output_data = (output_data.astype(np.float32) - output_zero_point) * output_scale

	return inference_ms, output_data


def main() -> int:
	"""Entry point for TFLite export and verification."""

	try:
		ensure_class_names_file()
		tflite_model = convert_to_tflite()
		save_tflite_model(tflite_model)

		model_size_kb = TFLITE_MODEL_PATH.stat().st_size / 1024.0
		inference_ms, output_data = run_tflite_verification()

		print(f"Saved TFLite model to: {TFLITE_MODEL_PATH}")
		print(f"Model size: {model_size_kb:.2f} KB")
		print(f"Inference time: {inference_ms:.2f} ms")
		print(f"Verification output shape: {output_data.shape}")
		print("TFLite export and verification completed successfully.")
		return 0

	except Exception as exc:  # noqa: BLE001 - intentional graceful top-level handling
		print_error(str(exc))
		return 1


if __name__ == "__main__":
	raise SystemExit(main())

