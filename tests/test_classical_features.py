"""Sanity checks for classical feature extraction against known-shape signals."""

from __future__ import annotations

import numpy as np

from tinyml_vibration.features.classical import FEATURE_NAMES, extract_features

FS = 12_000.0
N = 2048


def test_feature_vector_has_all_named_features() -> None:
    rng = np.random.default_rng(0)
    signal = rng.normal(size=N)
    features = extract_features(signal, FS)
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in features.values())


def test_sine_wave_rms_and_crest_factor() -> None:
    t = np.arange(N) / FS
    amplitude = 2.0
    signal = amplitude * np.sin(2 * np.pi * 100.0 * t)
    features = extract_features(signal, FS)
    expected_rms = amplitude / np.sqrt(2)
    assert abs(features["rms"] - expected_rms) / expected_rms < 0.01
    # Crest factor of a pure sine wave is sqrt(2) ~= 1.414.
    assert abs(features["crest_factor"] - np.sqrt(2)) < 0.02


def test_sine_wave_kurtosis_is_near_gaussian_reference() -> None:
    t = np.arange(N) / FS
    signal = np.sin(2 * np.pi * 100.0 * t)
    features = extract_features(signal, FS)
    # A sine wave's (Pearson, non-excess) kurtosis is 1.5, well below the
    # ~3.0 of white Gaussian noise and far below an impulsive fault signal.
    assert abs(features["kurtosis"] - 1.5) < 0.05


def test_impulsive_signal_has_high_kurtosis_and_crest_factor() -> None:
    rng = np.random.default_rng(1)
    signal = rng.normal(0, 0.01, size=N)
    signal[::128] += 5.0  # sparse large impulses, like a bearing fault impact train
    features = extract_features(signal, FS)
    assert features["kurtosis"] > 5.0
    assert features["crest_factor"] > 3.0


def test_dominant_frequency_is_detected() -> None:
    t = np.arange(N) / FS
    target_freq = 250.0
    signal = np.sin(2 * np.pi * target_freq * t)
    features = extract_features(signal, FS)
    resolution = FS / N
    assert abs(features["dominant_freq_hz"] - target_freq) <= resolution


def test_envelope_kurtosis_flags_amplitude_modulated_impacts() -> None:
    # A resonance carrier amplitude-modulated by a sparse impact train has a
    # much peakier envelope than the carrier alone -- the classic bearing
    # envelope-analysis signature.
    rng = np.random.default_rng(3)
    t = np.arange(N) / FS
    carrier = np.sin(2 * np.pi * 3000.0 * t) + rng.normal(0, 1e-3, size=N)
    modulator = np.zeros(N)
    modulator[::170] = 1.0
    envelope_signal = carrier * (0.05 + np.convolve(modulator, np.ones(20), mode="same"))

    plain_carrier_features = extract_features(carrier, FS)
    modulated_features = extract_features(envelope_signal, FS)
    assert modulated_features["envelope_kurtosis"] > plain_carrier_features["envelope_kurtosis"]


def test_band_energy_ratios_sum_to_one() -> None:
    rng = np.random.default_rng(2)
    signal = rng.normal(size=N)
    features = extract_features(signal, FS)
    band_sum = sum(v for k, v in features.items() if k.startswith("band_energy_ratio_"))
    assert abs(band_sum - 1.0) < 1e-6
