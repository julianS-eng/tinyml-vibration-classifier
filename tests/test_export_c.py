"""The C-array exporter must produce well-formed, parseable C source."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from tinyml_vibration.models.export_c import (
    tflite_to_c_source,
    write_c_source_files,
    write_test_vectors,
)


def test_tflite_to_c_source_round_trips_bytes() -> None:
    payload = bytes(range(256)) * 3  # exercise multiple lines, all byte values
    header, source = tflite_to_c_source(payload, var_name="g_model")

    assert "extern const unsigned char g_model[];" in header
    assert "#ifndef" in header and "#endif" in header

    hex_bytes = re.findall(r"0x([0-9a-f]{2})", source)
    recovered = bytes(int(b, 16) for b in hex_bytes)
    assert recovered == payload
    assert f"const unsigned int g_model_len = {len(payload)};" in source


def test_write_c_source_files_creates_both_files(tmp_path: Path) -> None:
    header_path, source_path = write_c_source_files(b"\x00\x01\x02\x03", tmp_path, "g_model")
    assert header_path.exists()
    assert source_path.exists()
    assert header_path.read_text().startswith("#ifndef")
    assert "0x00, 0x01, 0x02, 0x03" in source_path.read_text()


def test_write_test_vectors_shapes_and_labels(tmp_path: Path) -> None:
    windows = np.array([[1, -1, 2, -2], [3, -3, 4, -4]], dtype=np.int8)
    labels = ["normal", "imbalance"]
    header_path, source_path = write_test_vectors(windows, labels, tmp_path, "g_tv")

    header_text = header_path.read_text()
    assert "#define G_TV_COUNT 2" in header_text
    assert "#define G_TV_WINDOW_LENGTH 4" in header_text

    source_text = source_path.read_text()
    assert '"normal"' in source_text
    assert '"imbalance"' in source_text
    assert "1, -1, 2, -2" in source_text
