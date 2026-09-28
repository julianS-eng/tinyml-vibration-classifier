"""Common data interface so synthetic and real (e.g. CWRU) sources are swappable.

Any loader — :class:`~tinyml_vibration.data.synthetic.SyntheticBearingDataset`
or a real-data loader such as
:class:`~tinyml_vibration.data.cwru_loader.CWRUDataset` — implements
:class:`VibrationDataset` and yields :class:`VibrationSample` windows. The rest
of the pipeline (feature extraction, baseline model, CNN training) only
depends on this interface, so swapping synthetic data for real recordings
requires no changes downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True, slots=True)
class VibrationSample:
    """A single fixed-length vibration window with its label and metadata.

    Attributes:
        signal: 1D array of shape ``(window_length,)``, acceleration in g.
        label: Fault class name, one of ``tinyml_vibration.FAULT_CLASSES``.
        sampling_rate_hz: Sampling rate of ``signal`` in Hz.
        shaft_speed_hz: Shaft rotation frequency in Hz at acquisition time.
        source: Free-text provenance, e.g. ``"synthetic"`` or ``"cwru:1797rpm"``.
    """

    signal: npt.NDArray[np.float64]
    label: str
    sampling_rate_hz: float
    shaft_speed_hz: float
    source: str = "unknown"


class VibrationDataset(Protocol):
    """Interface every vibration data source (synthetic or real) must implement."""

    sampling_rate_hz: float

    def samples(self) -> list[VibrationSample]:
        """Return all windows in this dataset."""
        ...
