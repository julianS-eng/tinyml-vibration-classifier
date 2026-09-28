"""Classical-features + scikit-learn baseline classifier.

A Random Forest over the hand-engineered features from
:mod:`tinyml_vibration.features.classical`. This is the reference the 1D CNN
is compared against: cheap to train, easy to interpret (feature
importances), and a useful sanity check that the classes are separable
before investing in a neural network / on-device deployment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def build_baseline_pipeline(seed: int = 42, n_estimators: int = 200) -> Pipeline:
    """Build the scaler + Random Forest pipeline used as the classical baseline."""
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=None,
                    class_weight="balanced",
                    random_state=seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )


@dataclass(frozen=True, slots=True)
class CrossValidationResult:
    """Stratified k-fold cross-validation results for the baseline model."""

    accuracy_scores: FloatArray
    f1_macro_scores: FloatArray
    cv_predictions: IntArray

    @property
    def accuracy_mean(self) -> float:
        return float(np.mean(self.accuracy_scores))

    @property
    def accuracy_std(self) -> float:
        return float(np.std(self.accuracy_scores))

    @property
    def f1_macro_mean(self) -> float:
        return float(np.mean(self.f1_macro_scores))

    @property
    def f1_macro_std(self) -> float:
        return float(np.std(self.f1_macro_scores))


def cross_validate_baseline(
    features: FloatArray,
    labels: IntArray,
    n_splits: int = 5,
    seed: int = 42,
) -> CrossValidationResult:
    """Run stratified k-fold cross-validation of the baseline pipeline.

    Returns both per-fold accuracy/F1-macro scores and out-of-fold predictions
    (via ``cross_val_predict``), suitable for building an unbiased confusion
    matrix over the whole dataset.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pipeline = build_baseline_pipeline(seed=seed)

    accuracy_scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(pipeline, features, labels, cv=cv, scoring="f1_macro")
    cv_predictions = cross_val_predict(pipeline, features, labels, cv=cv)

    return CrossValidationResult(
        accuracy_scores=accuracy_scores,
        f1_macro_scores=f1_scores,
        cv_predictions=cv_predictions,
    )
