"""Synthetic and real (CWRU) loaders must be interchangeable."""

from __future__ import annotations

from pathlib import Path

import pytest

from tinyml_vibration.data.cwru_loader import CWRUDataset, _infer_label
from tinyml_vibration.data.loader import VibrationSample
from tinyml_vibration.data.synthetic import SyntheticBearingDataset


def test_synthetic_dataset_yields_vibration_samples() -> None:
    dataset = SyntheticBearingDataset(n_windows_per_class=2, seed=0)
    samples = dataset.samples()
    assert all(isinstance(s, VibrationSample) for s in samples)
    assert hasattr(dataset, "sampling_rate_hz")


@pytest.mark.parametrize(
    "filename,expected_label",
    [
        ("Normal_1.mat", "normal"),
        ("IR007_1.mat", "inner_race_fault"),
        ("OR007@6_1.mat", "outer_race_fault"),
        ("B021_2.mat", "imbalance"),
    ],
)
def test_infer_label_from_cwru_filename(filename: str, expected_label: str) -> None:
    assert _infer_label(filename) == expected_label


def test_infer_label_raises_on_unknown_prefix() -> None:
    with pytest.raises(ValueError, match="Cannot infer fault class"):
        _infer_label("mystery_file.mat")


def test_cwru_dataset_raises_clear_error_when_directory_empty(tmp_path: Path) -> None:
    dataset = CWRUDataset(data_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match=r"No \.mat files found"):
        dataset.samples()
