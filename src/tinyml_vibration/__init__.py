"""TinyML Vibration Classifier: predictive maintenance on microcontrollers.

Classifies rotating-machine vibration signals into ``normal``, ``imbalance``,
``inner_race_fault`` and ``outer_race_fault`` using classical DSP features and
a quantized 1D CNN small enough to run on an ESP32-S3.
"""

from __future__ import annotations

__version__ = "0.1.0"

FAULT_CLASSES: tuple[str, ...] = (
    "normal",
    "imbalance",
    "inner_race_fault",
    "outer_race_fault",
)
