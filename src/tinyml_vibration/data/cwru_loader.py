"""Loader for the real Case Western Reserve University (CWRU) bearing data set.

This is the "swappable" counterpart to
:class:`~tinyml_vibration.data.synthetic.SyntheticBearingDataset`: it
implements the same :class:`~tinyml_vibration.data.loader.VibrationDataset`
interface, so pointing the pipeline at real CWRU recordings instead of
synthetic data is a one-line change (see ``pipeline.py``).

.. note::
   **Not exercised in this repository's CI or in the results reported in the
   README.** The CWRU file server (``engineering.case.edu`` /
   ``csegroups.case.edu``) was not reachable from the network this project
   was built in (outbound access was blocked by policy), so this loader is
   implemented against CWRU's well-documented ``.mat`` file format and
   naming convention but has not been run against the real files. Treat it
   as a documented starting point — if a defect location, load, or fault
   diameter code is missing from :data:`_FAULT_LOCATION_CODES` for the
   files you have, extend the mapping rather than assuming the labels are
   wrong.

Expected data layout
---------------------
Download the 12 kHz drive-end ``.mat`` files from the CWRU Bearing Data
Center (https://engineering.case.edu/bearingdatacenter) into a single
directory, keeping the original filenames (e.g. ``IR007_1.mat``,
``OR007@6_1.mat``, ``B007_1.mat``, ``Normal_1.mat``). Each ``.mat`` file
contains a drive-end accelerometer channel named ``X<file_number>_DE_time``
and, for faulted bearings, an ``RPM`` variable.

Usage::

    from tinyml_vibration.data.cwru_loader import CWRUDataset

    dataset = CWRUDataset(data_dir="data/raw/cwru", window_length=2048)
    samples = dataset.samples()  # same VibrationSample objects as synthetic data
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tinyml_vibration.data.loader import VibrationSample

#: Best-effort mapping from CWRU filename prefixes to this project's fault
#: classes. CWRU encodes fault type (IR/OR/B), fault diameter in thousandths
#: of an inch, and for outer-race faults the clock position of the defect
#: relative to the load zone (e.g. ``@6``). We only need the fault type.
_FAULT_LOCATION_CODES: dict[str, str] = {
    "normal": "normal",
    "ir": "inner_race_fault",
    "or": "outer_race_fault",
    "b": "imbalance",  # ball fault: closest available class in this project's taxonomy
}

_FILENAME_RE = re.compile(r"^(?P<code>[a-zA-Z]+)", re.IGNORECASE)
_DEFAULT_RPM_HZ = 29.95  # CWRU nameplate ~1797 rpm, used when no RPM variable is present


def _infer_label(filename: str) -> str:
    stem = Path(filename).stem
    match = _FILENAME_RE.match(stem)
    code = (match.group("code") if match else stem).lower()
    if code not in _FAULT_LOCATION_CODES:
        msg = (
            f"Cannot infer fault class from CWRU filename '{filename}' "
            f"(parsed prefix '{code}'). Extend _FAULT_LOCATION_CODES in "
            "cwru_loader.py or rename the file to match the CWRU convention "
            "(Normal_*, IR*, OR*, B*)."
        )
        raise ValueError(msg)
    return _FAULT_LOCATION_CODES[code]


@dataclass(slots=True)
class CWRUDataset:
    """Load real CWRU drive-end ``.mat`` recordings as fixed-length windows.

    Attributes:
        data_dir: Directory containing CWRU ``.mat`` files.
        sampling_rate_hz: Sampling rate of the CWRU drive-end channel (12 kHz).
        window_length: Number of samples per window.
        hop_length: Stride between consecutive windows (defaults to
            ``window_length``, i.e. non-overlapping windows).
    """

    data_dir: str | Path
    sampling_rate_hz: float = 12_000.0
    window_length: int = 2048
    hop_length: int | None = None

    def samples(self) -> list[VibrationSample]:
        try:
            from scipy.io import loadmat
        except ImportError as exc:  # pragma: no cover - scipy is a core dependency
            msg = "scipy is required to load CWRU .mat files"
            raise ImportError(msg) from exc

        data_dir = Path(self.data_dir)
        mat_files = sorted(data_dir.glob("*.mat"))
        if not mat_files:
            msg = (
                f"No .mat files found in {data_dir}. Download the CWRU drive-end "
                "12 kHz dataset from https://engineering.case.edu/bearingdatacenter "
                "into this directory first."
            )
            raise FileNotFoundError(msg)

        hop = self.hop_length or self.window_length
        out: list[VibrationSample] = []
        for path in mat_files:
            label = _infer_label(path.name)
            mat = loadmat(path)
            de_key = next((k for k in mat if k.endswith("_DE_time")), None)
            if de_key is None:
                msg = f"No *_DE_time (drive-end) channel found in {path.name}"
                raise ValueError(msg)
            signal = np.asarray(mat[de_key], dtype=np.float64).reshape(-1)

            rpm_key = next((k for k in mat if k.endswith("RPM")), None)
            shaft_speed_hz = (
                float(np.asarray(mat[rpm_key]).reshape(-1)[0]) / 60.0
                if rpm_key is not None
                else _DEFAULT_RPM_HZ
            )

            for start in range(0, len(signal) - self.window_length + 1, hop):
                window = signal[start : start + self.window_length]
                out.append(
                    VibrationSample(
                        signal=window,
                        label=label,
                        sampling_rate_hz=self.sampling_rate_hz,
                        shaft_speed_hz=shaft_speed_hz,
                        source=f"cwru:{path.name}",
                    )
                )
        return out
