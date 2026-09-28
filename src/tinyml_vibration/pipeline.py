"""End-to-end pipeline: data -> features -> baseline -> CNN -> quantization -> export.

Running this module (``python -m tinyml_vibration.pipeline``) regenerates
every number, table and figure referenced by the README from scratch, using
only the fixed random seed below — nothing in the README is hand-typed or
estimated. See ``docs/results.json`` for the exact machine-readable output of
the last recorded run, and ``docs/img/`` for the figures it produced.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from tinyml_vibration import FAULT_CLASSES
from tinyml_vibration.data.bearing_physics import bpfi, bpfo
from tinyml_vibration.data.synthetic import SyntheticBearingDataset
from tinyml_vibration.evaluation.metrics import (
    f1_per_class,
    plot_confusion_matrix,
    plot_example_signals,
    plot_feature_importance,
    summarize_model_comparison,
)
from tinyml_vibration.features.classical import FEATURE_NAMES, extract_features
from tinyml_vibration.models.baseline import build_baseline_pipeline, cross_validate_baseline
from tinyml_vibration.models.cnn import build_cnn, train_cnn
from tinyml_vibration.models.export_c import write_c_source_files, write_test_vectors
from tinyml_vibration.models.quantize import (
    convert_to_tflite_float32,
    convert_to_tflite_int8,
    estimate_peak_activation_bytes,
    get_required_ops,
    model_size_kb,
    quantize_input,
    run_tflite_inference,
)

SEED = 42
WINDOW_LENGTH = 2048
N_WINDOWS_PER_CLASS = 300
CLASS_NAMES = list(FAULT_CLASSES)
LABEL_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = REPO_ROOT / "docs" / "img"
RESULTS_PATH = REPO_ROOT / "docs" / "results.json"
FIRMWARE_MODEL_DIR = REPO_ROOT / "firmware" / "esp32s3" / "main" / "model"


def _to_native(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays to plain Python, valid-JSON values."""
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.generic):
        return _to_native(obj.item())
    if isinstance(obj, np.ndarray):
        return _to_native(obj.tolist())
    if isinstance(obj, float) and np.isnan(obj):
        return None  # NaN is not valid JSON; e.g. baseline has no TFLite size_kb
    return obj


def build_feature_dataframe(samples: list[Any]) -> pd.DataFrame:
    """Extract classical features for every sample into a tidy DataFrame."""
    rows = []
    for sample in samples:
        row = extract_features(sample.signal, sample.sampling_rate_hz)
        row["label"] = sample.label
        row["label_index"] = LABEL_TO_INDEX[sample.label]
        row["shaft_speed_hz"] = sample.shaft_speed_hz
        rows.append(row)
    return pd.DataFrame(rows, columns=[*FEATURE_NAMES, "label", "label_index", "shaft_speed_hz"])


def run_pipeline(seed: int = SEED) -> dict[str, Any]:
    """Run the full pipeline and return a JSON-serializable results summary."""
    t_start = time.time()
    results: dict[str, Any] = {"seed": seed, "window_length": WINDOW_LENGTH}

    # -- 1. Data -----------------------------------------------------------
    dataset = SyntheticBearingDataset(
        window_length=WINDOW_LENGTH, n_windows_per_class=N_WINDOWS_PER_CLASS, seed=seed
    )
    samples = dataset.samples()
    results["n_samples"] = len(samples)
    results["classes"] = CLASS_NAMES

    features_df = build_feature_dataframe(samples)
    x_features = features_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = features_df["label_index"].to_numpy(dtype=np.int64)
    x_raw = np.stack([s.signal for s in samples]).astype(np.float32)

    # -- 2. Classical-features baseline (scikit-learn, cross-validated) ----
    cv_result = cross_validate_baseline(x_features, y, n_splits=5, seed=seed)
    baseline_cm_path = plot_confusion_matrix(
        y,
        cv_result.cv_predictions,
        CLASS_NAMES,
        FIGURES_DIR / "confusion_matrix_baseline.png",
        title="Baseline (Random Forest) - 5-fold CV confusion matrix",
    )
    baseline_f1 = f1_per_class(y, cv_result.cv_predictions, CLASS_NAMES)
    baseline_pipeline = build_baseline_pipeline(seed=seed)
    baseline_pipeline.fit(x_features, y)
    feature_importances = dict(
        sorted(
            zip(
                FEATURE_NAMES,
                (float(v) for v in baseline_pipeline["classifier"].feature_importances_),
                strict=True,
            ),
            key=lambda kv: kv[1],
            reverse=True,
        )
    )
    importance_fig_path = plot_feature_importance(
        feature_importances, FIGURES_DIR / "feature_importance.png", top_n=10
    )
    results["baseline"] = {
        "cv_accuracy_mean": cv_result.accuracy_mean,
        "cv_accuracy_std": cv_result.accuracy_std,
        "cv_f1_macro_mean": cv_result.f1_macro_mean,
        "cv_f1_macro_std": cv_result.f1_macro_std,
        "f1_per_class": baseline_f1,
        "confusion_matrix_figure": str(baseline_cm_path.relative_to(REPO_ROOT)),
        "top_features": dict(list(feature_importances.items())[:10]),
        "feature_importance_figure": str(importance_fig_path.relative_to(REPO_ROOT)),
    }

    # Illustrative signal panel: one example window per class, with the
    # inner/outer-race envelope-spectrum peak marked at its theoretical
    # BPFI/BPFO frequency -- visual evidence the synthetic generator is
    # physically grounded, not just labeled noise.
    example_by_class: dict[str, tuple[np.ndarray, float]] = {}
    fault_freqs: dict[str, float] = {}
    for sample in samples:
        if sample.label not in example_by_class:
            example_by_class[sample.label] = (sample.signal, sample.shaft_speed_hz)
        if len(example_by_class) == len(CLASS_NAMES):
            break
    for label, (_signal, shaft_hz) in example_by_class.items():
        if label == "inner_race_fault":
            fault_freqs[label] = bpfi(shaft_hz)
        elif label == "outer_race_fault":
            fault_freqs[label] = bpfo(shaft_hz)
    example_signals_path = plot_example_signals(
        {name: example_by_class[name] for name in CLASS_NAMES},
        sampling_rate_hz=dataset.sampling_rate_hz,
        output_path=FIGURES_DIR / "example_signals.png",
        fault_freq_hz=fault_freqs,
    )
    results["example_signals_figure"] = str(example_signals_path.relative_to(REPO_ROOT))

    # -- 3. Train/val/test split for the CNN (stratified, fixed seed) ------
    idx = np.arange(len(samples))
    idx_train, idx_temp = train_test_split(idx, test_size=0.4, stratify=y, random_state=seed)
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.5, stratify=y[idx_temp], random_state=seed
    )

    x_scale = float(np.percentile(np.abs(x_raw[idx_train]), 99))
    x_scale = x_scale if x_scale > 0 else 1.0

    def prep(indices: np.ndarray) -> np.ndarray:
        return (x_raw[indices] / x_scale)[..., np.newaxis]

    x_train, x_val, x_test = prep(idx_train), prep(idx_val), prep(idx_test)
    y_train, y_val, y_test = y[idx_train], y[idx_val], y[idx_test]

    results["split"] = {
        "n_train": len(idx_train),
        "n_val": len(idx_val),
        "n_test": len(idx_test),
        "raw_signal_scale": x_scale,
    }

    # -- 4. CNN (float32 Keras) ---------------------------------------------
    model = build_cnn(input_length=WINDOW_LENGTH, n_classes=len(CLASS_NAMES), seed=seed)
    history = train_cnn(model, x_train, y_train, x_val, y_val, seed=seed)
    results["cnn_training"] = {
        "epochs_run": len(history.history["loss"]),
        "final_val_accuracy": float(history.history["val_accuracy"][-1]),
    }

    y_pred_keras = np.argmax(model.predict(x_test, verbose=0), axis=1)
    keras_cm_path = plot_confusion_matrix(
        y_test,
        y_pred_keras,
        CLASS_NAMES,
        FIGURES_DIR / "confusion_matrix_cnn_float32.png",
        title="1D CNN (float32, Keras) - test-set confusion matrix",
    )
    keras_accuracy = float(np.mean(y_pred_keras == y_test))
    keras_f1 = f1_per_class(y_test, y_pred_keras, CLASS_NAMES)

    # -- 5. TFLite float32 vs. int8 quantized -------------------------------
    tflite_float = convert_to_tflite_float32(model)
    y_pred_float_tfl = run_tflite_inference(tflite_float, x_test)
    float_accuracy = float(np.mean(y_pred_float_tfl == y_test))

    tflite_int8 = convert_to_tflite_int8(model, x_train, seed=seed)
    y_pred_int8 = run_tflite_inference(tflite_int8, x_test)
    int8_accuracy = float(np.mean(y_pred_int8 == y_test))
    int8_f1 = f1_per_class(y_test, y_pred_int8, CLASS_NAMES)
    int8_cm_path = plot_confusion_matrix(
        y_test,
        y_pred_int8,
        CLASS_NAMES,
        FIGURES_DIR / "confusion_matrix_cnn_int8.png",
        title="1D CNN (int8, TFLite Micro-ready) - test-set confusion matrix",
    )

    ram_estimate_bytes = estimate_peak_activation_bytes(tflite_int8)
    comparison_df = summarize_model_comparison(
        [
            {
                "model": "Baseline (RF + classical features, 5-fold CV)",
                "accuracy": cv_result.accuracy_mean,
                "f1_macro": cv_result.f1_macro_mean,
                "size_kb": None,
                "ram_estimate_kb": None,
            },
            {
                "model": "CNN (float32, Keras)",
                "accuracy": keras_accuracy,
                "f1_macro": float(np.mean(list(keras_f1.values()))),
                "size_kb": None,
                "ram_estimate_kb": None,
            },
            {
                "model": "CNN (float32, TFLite)",
                "accuracy": float_accuracy,
                "f1_macro": None,
                "size_kb": model_size_kb(tflite_float),
                "ram_estimate_kb": None,
            },
            {
                "model": "CNN (int8, TFLite quantized)",
                "accuracy": int8_accuracy,
                "f1_macro": float(np.mean(list(int8_f1.values()))),
                "size_kb": model_size_kb(tflite_int8),
                "ram_estimate_kb": ram_estimate_bytes / 1024.0,
            },
        ]
    )

    results["cnn"] = {
        "keras_float32": {
            "test_accuracy": keras_accuracy,
            "f1_per_class": keras_f1,
            "confusion_matrix_figure": str(keras_cm_path.relative_to(REPO_ROOT)),
        },
        "tflite_float32": {
            "test_accuracy": float_accuracy,
            "size_kb": model_size_kb(tflite_float),
        },
        "tflite_int8": {
            "test_accuracy": int8_accuracy,
            "f1_per_class": int8_f1,
            "size_kb": model_size_kb(tflite_int8),
            "ram_estimate_kb": ram_estimate_bytes / 1024.0,
            "confusion_matrix_figure": str(int8_cm_path.relative_to(REPO_ROOT)),
        },
        "accuracy_drop_float_to_int8": float_accuracy - int8_accuracy,
        "size_reduction_ratio": model_size_kb(tflite_float) / model_size_kb(tflite_int8),
    }
    results["model_comparison_table"] = comparison_df.to_dict(orient="records")

    # -- 6. Robustness: harder noise + out-of-range speed ------------------
    robust_dataset = SyntheticBearingDataset(
        window_length=WINDOW_LENGTH,
        n_windows_per_class=100,
        rpm_range=(900.0, 4200.0),  # wider than training range (1200-3600 rpm)
        snr_db_range=(-3.0, 5.0),  # harsher noise than training range (3-20 dB)
        seed=seed + 1,
    )
    robust_samples = robust_dataset.samples()
    x_robust_raw = np.stack([s.signal for s in robust_samples]).astype(np.float32)
    y_robust = np.array([LABEL_TO_INDEX[s.label] for s in robust_samples], dtype=np.int64)
    x_robust = (x_robust_raw / x_scale)[..., np.newaxis]

    y_pred_robust_int8 = run_tflite_inference(tflite_int8, x_robust)
    robust_int8_accuracy = float(np.mean(y_pred_robust_int8 == y_robust))

    robust_features_df = build_feature_dataframe(robust_samples)
    x_robust_features = robust_features_df[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y_pred_robust_baseline = baseline_pipeline.predict(x_robust_features)
    robust_baseline_accuracy = float(np.mean(y_pred_robust_baseline == y_robust))

    results["robustness"] = {
        "description": (
            "Held-out stress set: rpm in [900, 4200] (training range [1200, 3600]) "
            "and SNR in [-3, 5] dB (training range [3, 20] dB)."
        ),
        "n_samples": len(robust_samples),
        "baseline_accuracy": robust_baseline_accuracy,
        "cnn_int8_accuracy": robust_int8_accuracy,
    }

    # -- 7. Export firmware artifacts ---------------------------------------
    header_path, source_path = write_c_source_files(
        tflite_int8, FIRMWARE_MODEL_DIR, var_name="g_vibration_model"
    )

    # One representative test-set window per class, pre-quantized, so the
    # firmware can self-test the model integration before a real sensor is
    # wired up (see firmware/esp32s3/main/inference.cc).
    test_vector_indices = [int(idx_test[y_test == i][0]) for i in range(len(CLASS_NAMES))]
    test_windows_raw = prep(np.array(test_vector_indices))
    test_windows_int8 = quantize_input(tflite_int8, test_windows_raw)[..., 0]
    tv_header_path, tv_source_path = write_test_vectors(
        test_windows_int8, CLASS_NAMES, FIRMWARE_MODEL_DIR, var_name="g_test_vectors"
    )

    results["firmware_export"] = {
        "header": str(header_path.relative_to(REPO_ROOT)),
        "source": str(source_path.relative_to(REPO_ROOT)),
        "model_bytes": len(tflite_int8),
        "test_vectors_header": str(tv_header_path.relative_to(REPO_ROOT)),
        "test_vectors_source": str(tv_source_path.relative_to(REPO_ROOT)),
        "required_tflite_ops": get_required_ops(tflite_int8),
    }

    results["runtime_seconds"] = time.time() - t_start
    return results


def main() -> None:
    """CLI entry point: run the pipeline and write ``docs/results.json``."""
    results = run_pipeline()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(_to_native(results), indent=2, sort_keys=False) + "\n")
    print(f"Wrote results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
