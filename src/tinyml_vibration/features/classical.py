"""Classical vibration-analysis features: time domain, FFT and envelope.

These are the hand-engineered features long used in industrial condition
monitoring (see e.g. Randall, "Vibration-based Condition Monitoring", Wiley
2011) and form the input to the scikit-learn baseline model in
:mod:`tinyml_vibration.models.baseline`. The 1D CNN (``models/cnn.py``)
instead learns directly from the raw windowed signal — the two approaches
are compared in the README.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.signal import hilbert
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import skew as _skew

FloatArray = npt.NDArray[np.float64]

_N_SPECTRAL_BANDS = 8

#: Ordered feature names, matching the dict keys returned by :func:`extract_features`.
FEATURE_NAMES: list[str] = [
    "rms",
    "peak",
    "peak_to_peak",
    "crest_factor",
    "kurtosis",
    "skewness",
    "shape_factor",
    "impulse_factor",
    "clearance_factor",
    "spectral_centroid_hz",
    "dominant_freq_hz",
    "dominant_freq_amplitude",
    "spectral_entropy",
    *[f"band_energy_ratio_{i}" for i in range(_N_SPECTRAL_BANDS)],
    "envelope_rms",
    "envelope_kurtosis",
    "envelope_dominant_freq_hz",
    "envelope_dominant_freq_amplitude",
]


def extract_features(signal: FloatArray, sampling_rate_hz: float) -> dict[str, float]:
    """Compute the full classical feature vector for one vibration window.

    Args:
        signal: 1D vibration signal (e.g. acceleration in g).
        sampling_rate_hz: Sampling rate of ``signal`` in Hz.

    Returns:
        Dict mapping each name in :data:`FEATURE_NAMES` to its value.
    """
    signal = np.asarray(signal, dtype=np.float64)
    features: dict[str, float] = {}
    features.update(_time_domain_features(signal))
    features.update(_frequency_domain_features(signal, sampling_rate_hz))
    features.update(_envelope_features(signal, sampling_rate_hz))
    return features


def _time_domain_features(x: FloatArray) -> dict[str, float]:
    rms = float(np.sqrt(np.mean(x**2)))
    peak = float(np.max(np.abs(x)))
    mean_abs = float(np.mean(np.abs(x)))
    mean_sqrt_abs = float(np.mean(np.sqrt(np.abs(x))))
    return {
        "rms": rms,
        "peak": peak,
        "peak_to_peak": float(np.max(x) - np.min(x)),
        "crest_factor": peak / rms if rms > 0 else 0.0,
        # Pearson's kurtosis (fisher=False): ~3 for a healthy Gaussian-like
        # signal, notably higher for impulsive bearing-fault signals.
        "kurtosis": float(_kurtosis(x, fisher=False)),
        "skewness": float(_skew(x)),
        "shape_factor": rms / mean_abs if mean_abs > 0 else 0.0,
        "impulse_factor": peak / mean_abs if mean_abs > 0 else 0.0,
        "clearance_factor": peak / mean_sqrt_abs**2 if mean_sqrt_abs > 0 else 0.0,
    }


def _frequency_domain_features(x: FloatArray, fs: float) -> dict[str, float]:
    n = len(x)
    windowed = x * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    # Ignore the DC bin for dominant-frequency / centroid purposes.
    ac_spectrum = spectrum[1:]
    ac_freqs = freqs[1:]
    total_energy = float(np.sum(ac_spectrum**2))

    if total_energy > 0:
        centroid = float(np.sum(ac_freqs * ac_spectrum) / np.sum(ac_spectrum))
        dominant_idx = int(np.argmax(ac_spectrum))
        dominant_freq = float(ac_freqs[dominant_idx])
        dominant_amp = float(ac_spectrum[dominant_idx])
        probs = ac_spectrum**2 / total_energy
        probs = probs[probs > 0]
        entropy = float(-np.sum(probs * np.log(probs)) / np.log(len(probs)))
    else:
        centroid = 0.0
        dominant_freq = 0.0
        dominant_amp = 0.0
        entropy = 0.0

    band_ratios = _band_energy_ratios(ac_freqs, ac_spectrum, total_energy, _N_SPECTRAL_BANDS)
    out = {
        "spectral_centroid_hz": centroid,
        "dominant_freq_hz": dominant_freq,
        "dominant_freq_amplitude": dominant_amp,
        "spectral_entropy": entropy,
    }
    for i, ratio in enumerate(band_ratios):
        out[f"band_energy_ratio_{i}"] = ratio
    return out


def _band_energy_ratios(
    freqs: FloatArray, spectrum: FloatArray, total_energy: float, n_bands: int
) -> list[float]:
    if total_energy == 0.0 or len(freqs) == 0:
        return [0.0] * n_bands
    nyquist = freqs[-1]
    edges = np.linspace(0.0, nyquist, n_bands + 1)
    ratios = []
    for i in range(n_bands):
        upper_inclusive = i == n_bands - 1  # capture the Nyquist bin in the last band
        if upper_inclusive:
            mask = (freqs >= edges[i]) & (freqs <= edges[i + 1])
        else:
            mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        band_energy = float(np.sum(spectrum[mask] ** 2))
        ratios.append(band_energy / total_energy)
    return ratios


def _envelope_features(x: FloatArray, fs: float) -> dict[str, float]:
    """Hilbert-transform envelope features (classic bearing-fault indicators).

    Bearing defect impacts are amplitude-modulated carriers of a structural
    resonance; demodulating with the analytic-signal envelope and looking at
    its spectrum recovers the (much lower) fault repetition frequency, which
    is otherwise buried under the resonance. High envelope kurtosis is a
    strong, well-established indicator of impulsive bearing faults.
    """
    envelope = np.abs(hilbert(x))
    envelope_rms = float(np.sqrt(np.mean(envelope**2)))
    envelope_kurt = float(_kurtosis(envelope, fisher=False))

    n = len(envelope)
    centered = envelope - np.mean(envelope)
    spectrum = np.abs(np.fft.rfft(centered * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    if len(spectrum) > 1 and np.max(spectrum[1:]) > 0:
        idx = int(np.argmax(spectrum[1:])) + 1
        env_dom_freq = float(freqs[idx])
        env_dom_amp = float(spectrum[idx])
    else:
        env_dom_freq = 0.0
        env_dom_amp = 0.0

    return {
        "envelope_rms": envelope_rms,
        "envelope_kurtosis": envelope_kurt,
        "envelope_dominant_freq_hz": env_dom_freq,
        "envelope_dominant_freq_amplitude": env_dom_amp,
    }
