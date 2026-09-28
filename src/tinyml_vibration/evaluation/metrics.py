"""Confusion matrices, per-class F1, and model comparison tables/figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: safe in CI and scripted runs
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

IntArray = npt.NDArray[np.int64]


def f1_per_class(y_true: IntArray, y_pred: IntArray, class_names: list[str]) -> dict[str, float]:
    """F1 score for each class, keyed by class name."""
    scores = f1_score(y_true, y_pred, average=None, labels=list(range(len(class_names))))
    return dict(zip(class_names, (float(s) for s in scores), strict=True))


def plot_confusion_matrix(
    y_true: IntArray,
    y_pred: IntArray,
    class_names: list[str],
    output_path: str | Path,
    title: str,
    normalize: bool = True,
) -> Path:
    """Render and save a confusion-matrix heatmap. Returns the saved path."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fmt = "d"
    display_cm: npt.NDArray[np.float64] | IntArray = cm
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        display_cm = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=np.float64), where=row_sums != 0
        )
        fmt = ".2f"

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        display_cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def summarize_model_comparison(records: list[dict[str, object]]) -> pd.DataFrame:
    """Build a tidy comparison table (accuracy, F1, size, RAM estimate) across models.

    Each record is expected to have keys: ``model``, ``accuracy``,
    ``f1_macro``, ``size_kb``, ``ram_estimate_kb`` (the last only applicable
    to TFLite models).
    """
    return pd.DataFrame.from_records(records)


def plot_feature_importance(
    importances: dict[str, float], output_path: str | Path, top_n: int = 10
) -> Path:
    """Bar chart of the top-``n`` Random Forest feature importances."""
    items = sorted(importances.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    names, values = zip(*items, strict=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, values, color=sns.color_palette("Blues_r", len(names)))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Random Forest feature importance")
    ax.set_title(f"Top {len(names)} classical features (baseline model)")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def plot_example_signals(
    samples_by_class: dict[str, tuple[npt.NDArray[np.float64], float]],
    sampling_rate_hz: float,
    output_path: str | Path,
    fault_freq_hz: dict[str, float] | None = None,
) -> Path:
    """Time-domain + envelope-spectrum panel, one column per class.

    Args:
        samples_by_class: ``{class_name: (signal, shaft_speed_hz)}``.
        sampling_rate_hz: Sampling rate shared by all signals.
        output_path: Where to save the figure.
        fault_freq_hz: Optional ``{class_name: expected_fault_freq_hz}`` to
            mark with a vertical line on the envelope-spectrum row (e.g.
            BPFI/BPFO), demonstrating the synthetic data's physical realism.
    """
    from scipy.signal import hilbert

    class_names = list(samples_by_class.keys())
    fig, axes = plt.subplots(2, len(class_names), figsize=(4 * len(class_names), 6), sharex="row")
    fault_freq_hz = fault_freq_hz or {}

    for col, name in enumerate(class_names):
        signal, _shaft_speed_hz = samples_by_class[name]
        n = len(signal)
        t = np.arange(n) / sampling_rate_hz

        ax_time = axes[0, col]
        ax_time.plot(t * 1000, signal, linewidth=0.6, color="#2b6cb0")
        ax_time.set_title(name)
        ax_time.set_xlabel("Time (ms)")
        if col == 0:
            ax_time.set_ylabel("Amplitude (g)")

        envelope = np.abs(hilbert(signal))
        envelope = envelope - envelope.mean()
        spectrum = np.abs(np.fft.rfft(envelope * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / sampling_rate_hz)

        ax_spec = axes[1, col]
        ax_spec.plot(freqs, spectrum, linewidth=0.7, color="#2f855a")
        ax_spec.set_xlim(0, 600)
        ax_spec.set_xlabel("Frequency (Hz)")
        if col == 0:
            ax_spec.set_ylabel("Envelope spectrum")
        if name in fault_freq_hz:
            ax_spec.axvline(fault_freq_hz[name], color="#c53030", linestyle="--", linewidth=1)
            ax_spec.text(
                fault_freq_hz[name],
                ax_spec.get_ylim()[1] * 0.9,
                " expected fault freq.",
                color="#c53030",
                fontsize=8,
            )

    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
