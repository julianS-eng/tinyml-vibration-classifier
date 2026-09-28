"""The synthetic generator must produce correctly shaped, reproducible data
whose fault classes actually show up at their theoretical BPFI/BPFO
frequencies -- otherwise "physically correct" would just be a docstring
claim.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import hilbert

from tinyml_vibration import FAULT_CLASSES
from tinyml_vibration.data.bearing_physics import bpfi, bpfo
from tinyml_vibration.data.synthetic import SyntheticBearingDataset


def test_dataset_shape_and_labels() -> None:
    dataset = SyntheticBearingDataset(n_windows_per_class=5, window_length=512, seed=0)
    samples = dataset.samples()
    assert len(samples) == 5 * len(FAULT_CLASSES)
    labels = {s.label for s in samples}
    assert labels == set(FAULT_CLASSES)
    assert all(s.signal.shape == (512,) for s in samples)
    assert all(np.isfinite(s.signal).all() for s in samples)


def test_dataset_is_reproducible_given_seed() -> None:
    a = SyntheticBearingDataset(n_windows_per_class=3, seed=123).samples()
    b = SyntheticBearingDataset(n_windows_per_class=3, seed=123).samples()
    for sa, sb in zip(a, b, strict=True):
        np.testing.assert_array_equal(sa.signal, sb.signal)
        assert sa.label == sb.label
        assert sa.shaft_speed_hz == sb.shaft_speed_hz


def test_different_seeds_give_different_signals() -> None:
    a = SyntheticBearingDataset(n_windows_per_class=3, seed=1).samples()
    b = SyntheticBearingDataset(n_windows_per_class=3, seed=2).samples()
    assert not np.array_equal(a[0].signal, b[0].signal)


@pytest.mark.parametrize("label,freq_fn", [("inner_race_fault", bpfi), ("outer_race_fault", bpfo)])
def test_fault_envelope_spectrum_peak_matches_bpfi_bpfo(label: str, freq_fn) -> None:
    dataset = SyntheticBearingDataset(
        n_windows_per_class=15, window_length=4096, snr_db_range=(20.0, 20.0), seed=5
    )
    samples = [s for s in dataset.samples() if s.label == label]
    relative_errors = []
    for sample in samples:
        fs = sample.sampling_rate_hz
        n = len(sample.signal)
        expected = freq_fn(sample.shaft_speed_hz)

        envelope = np.abs(hilbert(sample.signal))
        envelope = envelope - envelope.mean()
        spectrum = np.abs(np.fft.rfft(envelope * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)

        band = (freqs >= 0.5 * expected) & (freqs <= 1.5 * expected)
        peak_freq = freqs[band][np.argmax(spectrum[band])]
        relative_errors.append(abs(peak_freq - expected) / expected)

    # Envelope-spectrum peak should land within 5% of the theoretical
    # BPFI/BPFO frequency for the vast majority of windows.
    assert np.mean(relative_errors) < 0.05
    assert np.median(relative_errors) < 0.03


def test_imbalance_has_higher_1x_amplitude_than_normal() -> None:
    dataset = SyntheticBearingDataset(
        n_windows_per_class=10, window_length=4096, snr_db_range=(20.0, 20.0), seed=9
    )
    samples = dataset.samples()

    def dominant_1x_amplitude(signal: np.ndarray, fr: float, fs: float) -> float:
        n = len(signal)
        spectrum = np.abs(np.fft.rfft(signal * np.hanning(n)))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        idx = np.argmin(np.abs(freqs - fr))
        return float(spectrum[idx])

    normal_amps = [
        dominant_1x_amplitude(s.signal, s.shaft_speed_hz, s.sampling_rate_hz)
        for s in samples
        if s.label == "normal"
    ]
    imbalance_amps = [
        dominant_1x_amplitude(s.signal, s.shaft_speed_hz, s.sampling_rate_hz)
        for s in samples
        if s.label == "imbalance"
    ]
    assert np.mean(imbalance_amps) > np.mean(normal_amps)


def test_speed_and_snr_are_within_configured_ranges() -> None:
    rpm_range = (1500.0, 2500.0)
    snr_range = (5.0, 15.0)
    dataset = SyntheticBearingDataset(
        n_windows_per_class=10, rpm_range=rpm_range, snr_db_range=snr_range, seed=3
    )
    for sample in dataset.samples():
        rpm = sample.shaft_speed_hz * 60.0
        assert rpm_range[0] - 1e-6 <= rpm <= rpm_range[1] + 1e-6
