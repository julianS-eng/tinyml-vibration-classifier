"""Evaluation metrics and figures: confusion matrices, F1, size/RAM tables."""

from __future__ import annotations

from tinyml_vibration.evaluation.metrics import (
    f1_per_class,
    plot_confusion_matrix,
    summarize_model_comparison,
)

__all__ = ["f1_per_class", "plot_confusion_matrix", "summarize_model_comparison"]
