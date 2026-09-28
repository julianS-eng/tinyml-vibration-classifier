"""Physics-informed synthetic vibration data generator.

Why synthetic data
-------------------
This project targets the public Case Western Reserve University (CWRU)
bearing data set as its real-world reference, but CWRU's file server is not
reachable from every deployment environment (including the sandbox this
project was first built in — outbound access to ``engineering.case.edu`` was
blocked by network policy). Rather than silently ship without data, this
module generates synthetic vibration signals whose fault frequencies follow
the exact same rolling-element-bearing kinematics (BPFI/BPFO, see
:mod:`tinyml_vibration.data.bearing_physics`) that govern the real bearing.
The :class:`~tinyml_vibration.data.loader.VibrationDataset` interface means a
real loader (see :mod:`tinyml_vibration.data.cwru_loader`) is a drop-in
replacement for this one — nothing downstream needs to change.

Signal model
------------
Each window is built from physically motivated components:

* **normal**: low-amplitude 1x/2x/3x shaft harmonics (typical of a healthy,
  well-balanced machine) plus broadband noise.
* **imbalance**: a dominant 1x shaft-frequency sinusoid (mass unbalance
  produces a rotating force at exactly the shaft speed — the classic
  ISO 10816 imbalance signature) with weaker 2x/3x content.
* **inner_race_fault** / **outer_race_fault**: a train of impact-excited,
  exponentially decaying resonance pulses repeating at BPFI or BPFO
  respectively (an impact each time a rolling element strikes the race
  defect, ringing a structural resonance — the standard bearing-fault
  model, see Randall & Antoni 2011). The inner-race case is additionally
  amplitude-modulated at the shaft frequency because the defect rotates
  through the (fixed) load zone once per shaft revolution; the outer-race
  case is not modulated because the defect's position relative to the load
  zone is fixed.

Robustness axes
----------------
Shaft speed and signal-to-noise ratio are both randomized per window within
configurable ranges, so a model trained on this data is explicitly exercised
against speed variation and measurement noise rather than a single fixed
operating point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from tinyml_vibration.data.bearing_physics import BearingGeometry, bpfi, bpfo, rpm_to_hz
from tinyml_vibration.data.loader import VibrationSample

FloatArray = npt.NDArray[np.float64]


@dataclass(slots=True)
class SyntheticBearingDataset:
    """Generate synthetic normal/imbalance/inner-race/outer-race vibration windows.

    Attributes:
        sampling_rate_hz: Sampling rate in Hz (CWRU drive-end data uses 12 kHz).
        window_length: Number of samples per window.
        n_windows_per_class: Number of windows to generate for each fault class.
        rpm_range: Uniform sampling range for shaft speed, in rpm.
        snr_db_range: Uniform sampling range for signal-to-noise ratio, in dB.
        resonance_freq_hz: Structural resonance frequency excited by impacts.
        damping_ratio: Exponential decay rate of each impact response.
        seed: Base random seed for full reproducibility.
        geometry: Bearing geometry used to compute BPFI/BPFO.
    """

    sampling_rate_hz: float = 12_000.0
    window_length: int = 2048
    n_windows_per_class: int = 300
    rpm_range: tuple[float, float] = (1200.0, 3600.0)
    snr_db_range: tuple[float, float] = (3.0, 20.0)
    resonance_freq_hz: float = 3500.0
    damping_ratio: float = 700.0
    seed: int = 42
    geometry: BearingGeometry = field(default_factory=BearingGeometry)

    def samples(self) -> list[VibrationSample]:
        """Generate the full dataset (all classes, reproducible given ``seed``)."""
        rng = np.random.default_rng(self.seed)
        out: list[VibrationSample] = []
        generators = {
            "normal": self._normal,
            "imbalance": self._imbalance,
            "inner_race_fault": self._inner_race_fault,
            "outer_race_fault": self._outer_race_fault,
        }
        for label, gen in generators.items():
            for _ in range(self.n_windows_per_class):
                rpm = rng.uniform(*self.rpm_range)
                snr_db = rng.uniform(*self.snr_db_range)
                fr = rpm_to_hz(rpm)
                t = np.arange(self.window_length, dtype=np.float64) / self.sampling_rate_hz
                clean = gen(t, fr, rng)
                noisy = _add_noise(clean, snr_db, rng)
                out.append(
                    VibrationSample(
                        signal=noisy,
                        label=label,
                        sampling_rate_hz=self.sampling_rate_hz,
                        shaft_speed_hz=fr,
                        source=f"synthetic:rpm={rpm:.1f}:snr={snr_db:.1f}dB",
                    )
                )
        return out

    # -- per-class signal models -----------------------------------------

    def _shaft_harmonics(
        self,
        t: FloatArray,
        fr: float,
        rng: np.random.Generator,
        amplitudes: tuple[float, float, float],
    ) -> FloatArray:
        signal = np.zeros_like(t)
        for k, amp in enumerate(amplitudes, start=1):
            phase = rng.uniform(0.0, 2.0 * np.pi)
            signal += amp * np.sin(2.0 * np.pi * k * fr * t + phase)
        return signal

    def _normal(self, t: FloatArray, fr: float, rng: np.random.Generator) -> FloatArray:
        return self._shaft_harmonics(t, fr, rng, amplitudes=(0.05, 0.02, 0.01))

    def _imbalance(self, t: FloatArray, fr: float, rng: np.random.Generator) -> FloatArray:
        return self._shaft_harmonics(t, fr, rng, amplitudes=(0.35, 0.06, 0.02))

    def _inner_race_fault(self, t: FloatArray, fr: float, rng: np.random.Generator) -> FloatArray:
        baseline = self._shaft_harmonics(t, fr, rng, amplitudes=(0.05, 0.02, 0.01))
        fault_freq = bpfi(fr, self.geometry)
        pulses = _impact_pulse_train(
            t,
            fault_freq_hz=fault_freq,
            sampling_rate_hz=self.sampling_rate_hz,
            resonance_freq_hz=self.resonance_freq_hz,
            damping_ratio=self.damping_ratio,
            amplitude=1.0,
            rng=rng,
        )
        # Load-zone modulation: the inner-race defect rotates with the shaft,
        # so its impact amplitude is modulated once per shaft revolution.
        modulation = 1.0 + 0.6 * np.cos(2.0 * np.pi * fr * t + rng.uniform(0, 2 * np.pi))
        return baseline + pulses * modulation

    def _outer_race_fault(self, t: FloatArray, fr: float, rng: np.random.Generator) -> FloatArray:
        baseline = self._shaft_harmonics(t, fr, rng, amplitudes=(0.05, 0.02, 0.01))
        fault_freq = bpfo(fr, self.geometry)
        pulses = _impact_pulse_train(
            t,
            fault_freq_hz=fault_freq,
            sampling_rate_hz=self.sampling_rate_hz,
            resonance_freq_hz=self.resonance_freq_hz,
            damping_ratio=self.damping_ratio,
            amplitude=1.0,
            rng=rng,
        )
        # Outer-race defect position is fixed relative to the (fixed) load
        # zone, so amplitude is not modulated by shaft rotation.
        return baseline + pulses


def _impact_pulse_train(
    t: FloatArray,
    fault_freq_hz: float,
    sampling_rate_hz: float,
    resonance_freq_hz: float,
    damping_ratio: float,
    amplitude: float,
    rng: np.random.Generator,
    jitter_std_fraction: float = 0.02,
) -> FloatArray:
    """Sum of exponentially decaying sinusoids triggered at ``fault_freq_hz``.

    Models each rolling-element impact on a race defect as exciting a damped
    structural resonance (``resonance_freq_hz``). Impact timing is jittered
    slightly (``jitter_std_fraction`` of the fault period) to emulate the
    small amount of slip real bearings exhibit versus a perfectly periodic
    train.
    """
    period = 1.0 / fault_freq_hz
    duration = t[-1] - t[0] + 1.0 / sampling_rate_hz
    # Start a little before the window so decaying impacts triggered just
    # before t[0] still contribute (avoids an artificial silent lead-in).
    n_impacts = int(np.ceil(duration / period)) + 3
    impact_times = t[0] - 2.0 * period + period * np.arange(n_impacts)
    jitter = rng.normal(0.0, jitter_std_fraction * period, size=n_impacts)
    impact_times = impact_times + jitter

    signal = np.zeros_like(t)
    for t0 in impact_times:
        dt = t - t0
        mask = dt >= 0.0
        if not np.any(mask):
            continue
        envelope = np.zeros_like(t)
        envelope[mask] = amplitude * np.exp(-damping_ratio * dt[mask])
        signal += envelope * np.sin(2.0 * np.pi * resonance_freq_hz * dt)
    return signal


def _add_noise(signal: FloatArray, snr_db: float, rng: np.random.Generator) -> FloatArray:
    """Add white Gaussian noise to reach the target SNR (in dB)."""
    signal_power = float(np.mean(signal**2))
    if signal_power == 0.0:
        signal_power = 1e-12
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0.0, np.sqrt(noise_power), size=signal.shape)
    return signal + noise
