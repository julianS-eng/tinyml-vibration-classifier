"""Data loading: synthetic generator and a swappable real-data (CWRU) loader."""

from __future__ import annotations

from tinyml_vibration.data.loader import VibrationDataset, VibrationSample
from tinyml_vibration.data.synthetic import SyntheticBearingDataset

__all__ = ["SyntheticBearingDataset", "VibrationDataset", "VibrationSample"]
