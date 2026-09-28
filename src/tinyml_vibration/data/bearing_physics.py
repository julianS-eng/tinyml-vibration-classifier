"""Rolling-element bearing fault-frequency kinematics.

Standard formulas for a rolling-element bearing with the outer race fixed and
the inner race rotating with the shaft at frequency ``fr`` (Hz). See e.g.
Randall & Antoni, "Rolling element bearing diagnostics - A tutorial",
Mechanical Systems and Signal Processing 25 (2011).

    BPFO = (n / 2) * fr * (1 - (d / D) * cos(phi))   # ball pass freq., outer race
    BPFI = (n / 2) * fr * (1 + (d / D) * cos(phi))   # ball pass freq., inner race
    BSF  = (D / (2 * d)) * fr * (1 - (d / D)^2 * cos(phi)^2)  # ball spin frequency
    FTF  = (fr / 2) * (1 - (d / D) * cos(phi))       # fundamental train (cage) freq.

where ``n`` is the number of rolling elements, ``d`` the rolling-element
diameter, ``D`` the pitch diameter and ``phi`` the contact angle. Note
BPFO + BPFI == n * fr, which is a useful sanity check.

The default geometry below is the SKF 6205-2RS JEM deep-groove ball bearing
used at the drive end of the Case Western Reserve University (CWRU) bearing
data set test rig, so numbers here are directly comparable to that widely
used benchmark: at a shaft speed of 1797 rpm (29.95 Hz) this geometry gives
BPFO ~= 107.3 Hz and BPFI ~= 162.2 Hz, matching the values reported by CWRU
and in the bearing-diagnostics literature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class BearingGeometry:
    """Geometric parameters of a rolling-element bearing.

    Defaults are for the SKF 6205-2RS JEM deep-groove ball bearing (CWRU
    drive-end bearing): 9 balls, ball diameter 0.3126 in, pitch diameter
    1.537 in, contact angle 0 degrees.
    """

    n_elements: int = 9
    ball_diameter_mm: float = 7.94004  # 0.3126 in
    pitch_diameter_mm: float = 39.0398  # 1.537 in
    contact_angle_deg: float = 0.0

    @property
    def diameter_ratio(self) -> float:
        return self.ball_diameter_mm / self.pitch_diameter_mm

    @property
    def contact_angle_rad(self) -> float:
        return math.radians(self.contact_angle_deg)


def bpfo(shaft_freq_hz: float, geometry: BearingGeometry = BearingGeometry()) -> float:
    """Ball Pass Frequency, Outer race (Hz)."""
    ratio = geometry.diameter_ratio * math.cos(geometry.contact_angle_rad)
    return 0.5 * geometry.n_elements * shaft_freq_hz * (1.0 - ratio)


def bpfi(shaft_freq_hz: float, geometry: BearingGeometry = BearingGeometry()) -> float:
    """Ball Pass Frequency, Inner race (Hz)."""
    ratio = geometry.diameter_ratio * math.cos(geometry.contact_angle_rad)
    return 0.5 * geometry.n_elements * shaft_freq_hz * (1.0 + ratio)


def bsf(shaft_freq_hz: float, geometry: BearingGeometry = BearingGeometry()) -> float:
    """Ball Spin Frequency (Hz)."""
    ratio = geometry.diameter_ratio * math.cos(geometry.contact_angle_rad)
    return (
        (geometry.pitch_diameter_mm / (2.0 * geometry.ball_diameter_mm))
        * shaft_freq_hz
        * (1.0 - ratio**2)
    )


def ftf(shaft_freq_hz: float, geometry: BearingGeometry = BearingGeometry()) -> float:
    """Fundamental Train (cage) Frequency (Hz)."""
    ratio = geometry.diameter_ratio * math.cos(geometry.contact_angle_rad)
    return 0.5 * shaft_freq_hz * (1.0 - ratio)


def rpm_to_hz(rpm: float) -> float:
    """Convert shaft speed in rpm to shaft rotation frequency in Hz."""
    return rpm / 60.0
