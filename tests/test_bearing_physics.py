"""Fault-frequency formulas against the well-known CWRU 6205-2RS JEM geometry."""

from __future__ import annotations

from tinyml_vibration.data.bearing_physics import BearingGeometry, bpfi, bpfo, bsf, ftf, rpm_to_hz


def test_rpm_to_hz() -> None:
    assert abs(rpm_to_hz(1797.0) - 29.95) < 0.01


def test_bpfo_matches_cwru_reference_value() -> None:
    # CWRU / bearing-diagnostics literature: at 1797 rpm, BPFO ~= 107.3 Hz
    # for the 6205-2RS JEM drive-end bearing (multiplier ~3.585 * fr).
    fr = rpm_to_hz(1797.0)
    assert abs(bpfo(fr) - 107.3) < 0.5


def test_bpfi_matches_cwru_reference_value() -> None:
    # Same bearing: BPFI ~= 162.2 Hz at 1797 rpm (multiplier ~5.415 * fr).
    fr = rpm_to_hz(1797.0)
    assert abs(bpfi(fr) - 162.2) < 0.5


def test_bpfo_plus_bpfi_equals_n_times_shaft_freq() -> None:
    # Kinematic identity: BPFO + BPFI == n_elements * fr, independent of geometry.
    geometry = BearingGeometry()
    fr = 42.0
    total = bpfo(fr, geometry) + bpfi(fr, geometry)
    assert abs(total - geometry.n_elements * fr) < 1e-9


def test_ftf_and_bsf_are_positive_and_below_shaft_freq_family() -> None:
    fr = 30.0
    assert 0.0 < ftf(fr) < fr
    assert bsf(fr) > 0.0
