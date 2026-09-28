"""Convert the Keras CNN to TFLite (float32) and to fully int8-quantized TFLite.

Also provides model-size and a RAM (activation-arena) *estimate* used in the
README size/accuracy comparison. Int8 quantization approximately quarters
flash/RAM footprint versus float32 at a small accuracy cost — both are
measured in this module rather than assumed.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt
import tensorflow as tf
from tensorflow import keras

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def convert_to_tflite_float32(model: keras.Model) -> bytes:
    """Convert a Keras model to a plain (unquantized) float32 TFLite model."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    return converter.convert()


def convert_to_tflite_int8(
    model: keras.Model,
    representative_data: FloatArray,
    n_calibration_samples: int = 200,
    seed: int = 42,
) -> bytes:
    """Fully quantize a Keras model to int8 (weights, activations, I/O).

    Args:
        model: Trained Keras model with a single ``(batch, length, 1)`` input.
        representative_data: Float32 array of shape ``(n, length, 1)`` used to
            calibrate activation quantization ranges. Should be drawn from the
            training distribution (not the test set).
        n_calibration_samples: Number of windows from ``representative_data``
            to feed the converter's calibration pass.
        seed: Seed for subsampling ``representative_data``.
    """
    rng = np.random.default_rng(seed)
    n = min(n_calibration_samples, len(representative_data))
    idx = rng.choice(len(representative_data), size=n, replace=False)
    calibration_subset = representative_data[idx].astype(np.float32)

    def representative_dataset() -> Iterator[list[npt.NDArray[np.float32]]]:
        for i in range(len(calibration_subset)):
            yield [calibration_subset[i : i + 1]]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def run_tflite_inference(tflite_model: bytes, x: FloatArray) -> IntArray:
    """Run a TFLite model (float32 or int8) over ``x`` and return class predictions.

    Handles input/output quantization transparently: if the model expects
    int8 input, ``x`` (float32) is affine-quantized using the model's own
    scale/zero-point before inference. Argmax over an affinely-quantized
    (positive-scale) int8 output equals argmax over the dequantized
    probabilities, so predictions are taken directly on the raw output.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.empty(len(x), dtype=np.int64)
    scale, zero_point = input_details.get("quantization", (0.0, 0))
    for i in range(len(x)):
        sample = x[i : i + 1]
        if input_details["dtype"] == np.int8:
            sample = np.round(sample / scale + zero_point).astype(np.int8)
        else:
            sample = sample.astype(np.float32)
        interpreter.set_tensor(input_details["index"], sample)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        predictions[i] = int(np.argmax(output))
    return predictions


def quantize_input(tflite_model: bytes, x: FloatArray) -> npt.NDArray[np.int8]:
    """Affine-quantize float32 windows to the model's int8 input representation.

    Useful for exporting fixed test vectors (see
    :func:`tinyml_vibration.models.export_c.write_test_vectors`) that a
    device without floating-point conversion code can feed straight into the
    TFLite Micro interpreter.
    """
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    scale, zero_point = input_details["quantization"]
    quantized = np.round(x / scale + zero_point)
    return np.clip(quantized, -128, 127).astype(np.int8)


def _undelegated_interpreter(tflite_model: bytes) -> tf.lite.Interpreter:
    """Interpreter with default delegates (e.g. XNNPACK) disabled.

    Desktop TFLite applies the XNNPACK delegate by default, which fuses
    whole op subgraphs into an opaque ``DELEGATE`` node — great for desktop
    inference speed, but it hides the real per-op graph structure that
    ``get_required_ops`` and ``estimate_peak_activation_bytes`` need to
    inspect (TFLite Micro on the MCU has no such delegate).
    """
    interpreter = tf.lite.Interpreter(
        model_content=tflite_model,
        experimental_op_resolver_type=(
            tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
        ),
    )
    interpreter.allocate_tensors()
    return interpreter


def get_required_ops(tflite_model: bytes) -> list[str]:
    """List the TFLite builtin op names this model needs (for the MCU op resolver)."""
    interpreter = _undelegated_interpreter(tflite_model)
    try:
        ops = interpreter._get_ops_details()
    except AttributeError:
        return []
    seen = sorted({op["op_name"] for op in ops})
    return seen


def model_size_kb(tflite_model: bytes) -> float:
    """Model size in KiB (as it would be stored in flash)."""
    return len(tflite_model) / 1024.0


def estimate_peak_activation_bytes(tflite_model: bytes) -> int:
    """Estimate peak activation ("tensor arena") memory in bytes.

    This is a **lower-bound estimate**, not a measurement: for each op in
    execution order it sums the sizes of that op's activation tensors
    (inputs + outputs that are not constants), and reports the largest such
    sum across all ops. For a purely sequential feed-forward network (true
    of this project's CNN — no residual/skip connections) this closely
    tracks the true double-buffering requirement, but it does not model
    TFLite Micro's own interpreter/allocator bookkeeping overhead (typically
    a few hundred bytes to a few KB) or memory reuse across non-adjacent ops.
    Treat the reported value as a planning number; validate the real arena
    size on-device with ``tflite::RecordingMicroAllocator`` before finalizing
    a production memory budget (see ``firmware/esp32s3``).
    """
    interpreter = _undelegated_interpreter(tflite_model)
    try:
        ops = interpreter._get_ops_details()
    except AttributeError:
        return _fallback_activation_estimate(interpreter)

    tensor_details = {d["index"]: d for d in interpreter.get_tensor_details()}
    producer_indices = {out for op in ops for out in op["outputs"]}
    graph_input_indices = {d["index"] for d in interpreter.get_input_details()}
    activation_indices = producer_indices | graph_input_indices

    def tensor_bytes(idx: int) -> int:
        details = tensor_details[idx]
        n_elements = int(np.prod(details["shape"])) if len(details["shape"]) else 1
        return n_elements * np.dtype(details["dtype"]).itemsize

    peak = 0
    for op in ops:
        involved = list(op["inputs"]) + list(op["outputs"])
        op_bytes = sum(tensor_bytes(idx) for idx in involved if idx in activation_indices)
        peak = max(peak, op_bytes)
    return peak


def _fallback_activation_estimate(interpreter: tf.lite.Interpreter) -> int:
    """Fallback when the private ops-detail API is unavailable: 2x largest tensor."""
    sizes = [
        int(np.prod(d["shape"])) * np.dtype(d["dtype"]).itemsize
        for d in interpreter.get_tensor_details()
    ]
    sizes.sort(reverse=True)
    return sum(sizes[:2])
