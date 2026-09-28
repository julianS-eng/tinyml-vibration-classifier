"""Fast smoke tests for the baseline and CNN model paths.

Deliberately tiny (short windows, few samples, 2 epochs) so CI stays quick;
the real accuracy/size numbers reported in the README come from the full
run of `python -m tinyml_vibration.pipeline`, not from these tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from tinyml_vibration.data.synthetic import SyntheticBearingDataset
from tinyml_vibration.features.classical import FEATURE_NAMES, extract_features
from tinyml_vibration.models.baseline import build_baseline_pipeline, cross_validate_baseline
from tinyml_vibration.models.cnn import build_cnn, train_cnn
from tinyml_vibration.models.export_c import write_test_vectors
from tinyml_vibration.models.quantize import (
    convert_to_tflite_float32,
    convert_to_tflite_int8,
    estimate_peak_activation_bytes,
    get_required_ops,
    model_size_kb,
    quantize_input,
    run_tflite_inference,
)

WINDOW_LENGTH = 64
N_PER_CLASS = 12
SEED = 0


def _tiny_dataset():
    dataset = SyntheticBearingDataset(
        window_length=WINDOW_LENGTH, n_windows_per_class=N_PER_CLASS, seed=SEED
    )
    return dataset.samples()


def test_baseline_pipeline_fits_and_predicts() -> None:
    samples = _tiny_dataset()
    x = np.array([list(extract_features(s.signal, s.sampling_rate_hz).values()) for s in samples])
    labels = sorted({s.label for s in samples})
    y = np.array([labels.index(s.label) for s in samples])

    pipeline = build_baseline_pipeline(seed=SEED, n_estimators=20)
    pipeline.fit(x, y)
    predictions = pipeline.predict(x)
    assert predictions.shape == y.shape
    assert set(predictions).issubset(set(y))


def test_cross_validate_baseline_returns_scores_in_valid_range() -> None:
    samples = _tiny_dataset()
    x = np.array([list(extract_features(s.signal, s.sampling_rate_hz).values()) for s in samples])
    labels = sorted({s.label for s in samples})
    y = np.array([labels.index(s.label) for s in samples])

    result = cross_validate_baseline(x, y, n_splits=3, seed=SEED)
    assert len(FEATURE_NAMES) == x.shape[1]
    assert 0.0 <= result.accuracy_mean <= 1.0
    assert 0.0 <= result.f1_macro_mean <= 1.0
    assert result.cv_predictions.shape == y.shape


def test_cnn_train_quantize_and_export_round_trip(tmp_path: Path) -> None:
    samples = _tiny_dataset()
    x_raw = np.stack([s.signal for s in samples]).astype(np.float32)
    labels = sorted({s.label for s in samples})
    y = np.array([labels.index(s.label) for s in samples])

    scale = float(np.max(np.abs(x_raw))) or 1.0
    x = (x_raw / scale)[..., np.newaxis]

    model = build_cnn(input_length=WINDOW_LENGTH, n_classes=len(labels), seed=SEED)
    history = train_cnn(model, x, y, x, y, epochs=2, batch_size=8, seed=SEED)
    assert "loss" in history.history

    tflite_float = convert_to_tflite_float32(model)
    float_preds = run_tflite_inference(tflite_float, x)
    assert float_preds.shape == y.shape

    tflite_int8 = convert_to_tflite_int8(model, x, n_calibration_samples=10, seed=SEED)
    int8_preds = run_tflite_inference(tflite_int8, x)
    assert int8_preds.shape == y.shape

    # int8 model must not be larger than the float32 one.
    assert model_size_kb(tflite_int8) <= model_size_kb(tflite_float)

    ram_estimate = estimate_peak_activation_bytes(tflite_int8)
    assert ram_estimate > 0

    required_ops = get_required_ops(tflite_int8)
    assert isinstance(required_ops, list)

    quantized_input = quantize_input(tflite_int8, x[:2])
    assert quantized_input.dtype == np.int8
    assert quantized_input.shape == x[:2].shape

    write_test_vectors(
        quantized_input[..., 0], labels[:2], output_dir=tmp_path, var_name="g_smoke_test_vectors"
    )
