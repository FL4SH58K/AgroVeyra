"""Train a plant disease classifier using EfficientNetB0.

Expected dataset layout:

data/plant_disease/
├── train/
│   ├── class_a/
│   └── class_b/
└── valid/
	├── class_a/
	└── class_b/

The script performs two-stage training:
1. Train a classifier head with the EfficientNetB0 backbone frozen.
2. Fine-tune the last 30 layers of the backbone with a lower learning rate.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report


DATASET_PATH = "data/plant_disease"
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
MAX_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
INITIAL_LEARNING_RATE = 0.001
FINE_TUNE_LEARNING_RATE = 0.0001
MODEL_DIR = Path("models")
BEST_MODEL_PATH = MODEL_DIR / "best_model.keras"
FINAL_MODEL_PATH = MODEL_DIR / "final_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
PLOTS_PATH = MODEL_DIR / "training_plots.png"


def ensure_dataset_exists() -> tuple[Path, Path]:
	"""Validate the dataset directories before training starts."""

	dataset_root = Path(DATASET_PATH)
	train_dir = dataset_root / "train"
	valid_dir = dataset_root / "valid"

	if not train_dir.exists():
		raise FileNotFoundError(f"Training directory not found: {train_dir}")
	if not valid_dir.exists():
		raise FileNotFoundError(f"Validation directory not found: {valid_dir}")

	return train_dir, valid_dir


def build_datasets(train_dir: Path, valid_dir: Path) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str]]:
	"""Build efficient tf.data pipelines for training and validation."""

	train_ds = tf.keras.utils.image_dataset_from_directory(
		train_dir,
		image_size=IMAGE_SIZE,
		batch_size=BATCH_SIZE,
		label_mode="categorical",
		shuffle=True,
	)
	valid_ds = tf.keras.utils.image_dataset_from_directory(
		valid_dir,
		image_size=IMAGE_SIZE,
		batch_size=BATCH_SIZE,
		label_mode="categorical",
		shuffle=False,
	)

	class_names = train_ds.class_names

	autotune = tf.data.AUTOTUNE
	train_ds = train_ds.map(
		lambda images, labels: (
			tf.keras.applications.efficientnet.preprocess_input(tf.cast(images, tf.float32)),
			labels,
		),
		num_parallel_calls=autotune,
	)
	valid_ds = valid_ds.map(
		lambda images, labels: (
			tf.keras.applications.efficientnet.preprocess_input(tf.cast(images, tf.float32)),
			labels,
		),
		num_parallel_calls=autotune,
	)

	train_ds = train_ds.cache().prefetch(autotune)
	valid_ds = valid_ds.cache().prefetch(autotune)

	return train_ds, valid_ds, class_names


def build_model(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
	"""Create the EfficientNetB0 transfer-learning model."""

	inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
	base_model = tf.keras.applications.EfficientNetB0(
		include_top=False,
		weights="imagenet",
		input_tensor=inputs,
	)
	base_model.trainable = False

	x = base_model.output
	x = tf.keras.layers.GlobalAveragePooling2D()(x)
	x = tf.keras.layers.Dense(512, activation="relu")(x)
	x = tf.keras.layers.Dropout(0.4)(x)
	outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

	model = tf.keras.Model(inputs=inputs, outputs=outputs, name="agroveyra_plant_disease_classifier")
	return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
	"""Compile the model with the required optimizer and loss."""

	model.compile(
		optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss="categorical_crossentropy",
		metrics=["accuracy"],
	)


def build_callbacks() -> list[tf.keras.callbacks.Callback]:
	"""Create the callbacks requested for robust training."""

	MODEL_DIR.mkdir(parents=True, exist_ok=True)

	return [
		tf.keras.callbacks.EarlyStopping(
			monitor="val_loss",
			patience=5,
			restore_best_weights=True,
		),
		tf.keras.callbacks.ModelCheckpoint(
			filepath=str(BEST_MODEL_PATH),
			monitor="val_loss",
			save_best_only=True,
		),
		tf.keras.callbacks.ReduceLROnPlateau(
			monitor="val_loss",
			factor=0.5,
			patience=3,
		),
	]


def merge_histories(*histories: tf.keras.callbacks.History) -> dict[str, list[float]]:
	"""Combine training histories so plots show the full training run."""

	merged: dict[str, list[float]] = {}
	for history in histories:
		for key, values in history.history.items():
			merged.setdefault(key, []).extend(values)
	return merged


def plot_training_curves(history: dict[str, list[float]]) -> None:
	"""Save accuracy and loss curves to disk."""

	epochs = range(1, len(history.get("accuracy", [])) + 1)

	plt.figure(figsize=(12, 5))

	plt.subplot(1, 2, 1)
	plt.plot(epochs, history.get("accuracy", []), label="Training Accuracy")
	plt.plot(epochs, history.get("val_accuracy", []), label="Validation Accuracy")
	plt.xlabel("Epoch")
	plt.ylabel("Accuracy")
	plt.title("Accuracy Curves")
	plt.legend()

	plt.subplot(1, 2, 2)
	plt.plot(epochs, history.get("loss", []), label="Training Loss")
	plt.plot(epochs, history.get("val_loss", []), label="Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title("Loss Curves")
	plt.legend()

	plt.tight_layout()
	plt.savefig(PLOTS_PATH, dpi=200, bbox_inches="tight")
	plt.close()


def save_class_names(class_names: list[str]) -> None:
	"""Persist class labels for downstream inference."""

	with CLASS_NAMES_PATH.open("w", encoding="utf-8") as file_handle:
		json.dump(class_names, file_handle, indent=2)


def unfreeze_last_layers(base_model: tf.keras.Model, layers_to_unfreeze: int = 30) -> None:
	"""Unfreeze the last N layers of the backbone for fine-tuning."""

	base_model.trainable = True
	if layers_to_unfreeze <= 0:
		return

	for layer in base_model.layers[:-layers_to_unfreeze]:
		layer.trainable = False


def main() -> None:
	"""Train, fine-tune, evaluate, and persist the final classifier."""

	tf.keras.utils.set_random_seed(42)

	train_dir, valid_dir = ensure_dataset_exists()
	train_ds, valid_ds, class_names = build_datasets(train_dir, valid_dir)
	num_classes = len(class_names)

	if num_classes < 2:
		raise ValueError("The dataset must contain at least two classes.")

	save_class_names(class_names)

	model, base_model = build_model(num_classes)
	callbacks = build_callbacks()

	compile_model(model, INITIAL_LEARNING_RATE)
	initial_history = model.fit(
		train_ds,
		validation_data=valid_ds,
		epochs=MAX_EPOCHS,
		callbacks=callbacks,
	)

	unfreeze_last_layers(base_model, layers_to_unfreeze=30)
	compile_model(model, FINE_TUNE_LEARNING_RATE)

	fine_tune_history = model.fit(
		train_ds,
		validation_data=valid_ds,
		epochs=FINE_TUNE_EPOCHS,
		callbacks=callbacks,
	)

	model.save(FINAL_MODEL_PATH)

	merged_history = merge_histories(initial_history, fine_tune_history)
	plot_training_curves(merged_history)

	_, train_accuracy = model.evaluate(train_ds, verbose=0)
	_, valid_accuracy = model.evaluate(valid_ds, verbose=0)

	print(f"Final training accuracy: {train_accuracy:.4f}")
	print(f"Final validation accuracy: {valid_accuracy:.4f}")

	y_true = np.concatenate([labels.numpy() for _, labels in valid_ds], axis=0)
	y_probabilities = model.predict(valid_ds, verbose=0)
	y_pred = np.argmax(y_probabilities, axis=1)
	y_true_labels = np.argmax(y_true, axis=1)

	print("Classification report on validation set:")
	print(
		classification_report(
			y_true_labels,
			y_pred,
			labels=list(range(num_classes)),
			target_names=class_names,
			digits=4,
			zero_division=0,
		)
	)

	print(f"Best model saved to: {BEST_MODEL_PATH}")
	print(f"Final model saved to: {FINAL_MODEL_PATH}")
	print(f"Training plots saved to: {PLOTS_PATH}")


if __name__ == "__main__":
	main()

