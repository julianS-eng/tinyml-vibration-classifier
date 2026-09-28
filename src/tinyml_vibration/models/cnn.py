"""Small 1D CNN for on-device (ESP32-S3) vibration classification.

Learns directly from the raw windowed signal (no hand-engineered features),
trading interpretability for a model small enough to quantize to int8 and
run in a few KB of flash/RAM on a microcontroller. Compared against the
classical-features baseline (:mod:`tinyml_vibration.models.baseline`) in the
README.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def set_global_seed(seed: int) -> None:
    """Seed numpy, python-random and TensorFlow for reproducible training."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def build_cnn(input_length: int, n_classes: int, seed: int = 42) -> keras.Model:
    """Build the 1D CNN architecture.

    Three strided Conv1D blocks (progressively fewer time steps, more
    channels) followed by global average pooling and a small dense head.
    Global average pooling (rather than Flatten + large Dense) keeps the
    parameter count, and therefore the quantized model size, small.
    """
    set_global_seed(seed)
    inputs = keras.Input(shape=(input_length, 1), name="vibration_window")

    x = keras.layers.Conv1D(8, kernel_size=9, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(16, kernel_size=5, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.Conv1D(32, kernel_size=3, strides=2, padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.ReLU()(x)

    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dropout(0.3, seed=seed)(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="tiny_vibration_cnn")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_cnn(
    model: keras.Model,
    x_train: FloatArray,
    y_train: IntArray,
    x_val: FloatArray,
    y_val: IntArray,
    epochs: int = 60,
    batch_size: int = 32,
    seed: int = 42,
) -> keras.callbacks.History:
    """Train ``model`` with early stopping on validation loss."""
    set_global_seed(seed)
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    return model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
