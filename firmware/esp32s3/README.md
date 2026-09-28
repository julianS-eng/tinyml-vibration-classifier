# ESP32-S3 firmware skeleton

ESP-IDF project that embeds the int8-quantized TFLite model (via
[esp-tflite-micro](https://github.com/espressif/esp-tflite-micro)) and runs
an on-device self-test at boot using pre-quantized sample windows — no
accelerometer needs to be wired up to validate the model integration itself.

`model/g_vibration_model.{h,cc}` and `model/g_test_vectors.{h,cc}` are
**generated**, not hand-written: they come from
`python -m tinyml_vibration.pipeline` (see
`tinyml_vibration.models.export_c`) and are regenerated every time the
pipeline runs, so the bytes on the MCU are always exactly the bytes that
were evaluated on the test set in `docs/results.json`.

## Build

Requires the [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/get-started/index.html)
toolchain for `esp32s3` (v5.1+):

```bash
. $IDF_PATH/export.sh
cd firmware/esp32s3
idf.py set-target esp32s3
idf.py build
```

The first `idf.py build` downloads `espressif/esp-tflite-micro` from the
[ESP Component Registry](https://components.espressif.com/) per
`main/idf_component.yml` — this needs outbound network access to
`components.espressif.com` (see the note in the top-level README about this
being unreachable from the sandbox this project was authored in; CI builds
it on GitHub-hosted runners instead, which do have that access).

## Flash and monitor

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

Expect boot-time log lines like:

```
I (312) vibration_inference: Model ready. Arena used: <N> / 40960 bytes
I (315) main: Running embedded self-test (no sensor required)...
I (320) vibration_inference: Self-test 0: expected=normal predicted=normal confidence=0.9x latency=<N>us
...
W (xxx) main: No accelerometer driver wired up yet -- see README 'Future work'.
```

## Memory budget

* **Flash**: the model itself is `firmware/esp32s3/main/model/g_vibration_model.cc`'s
  byte-array size (see `docs/results.json` -> `cnn.tflite_int8.size_kb`), plus
  the esp-tflite-micro runtime and this application (typically well under
  200 KB total on top of the IDF/FreeRTOS base image — verify with
  `idf.py size` after building).
* **RAM (tensor arena)**: `kTensorArenaSize` in `main/inference.cc` is set
  from this project's *estimated* peak activation memory
  (`docs/results.json` -> `cnn.tflite_int8.ram_estimate_kb`) plus headroom.
  That estimate is a planning number, not a hardware measurement (see the
  docstring on `tinyml_vibration.models.quantize.estimate_peak_activation_bytes`);
  `InitModel()` logs the *real* arena usage
  (`MicroInterpreter::arena_used_bytes()`) at boot, which is the number to
  trust and to eventually replace `kTensorArenaSize` with (shrunk to that
  measured value plus a small safety margin) once you've flashed real
  hardware.

## Future work (explicitly out of scope here)

* **Accelerometer driver.** `main.cc` stubs this out with a `TODO`. Not
  implemented because the right driver depends on a hardware choice this
  repository doesn't make for you (e.g. ICM-42688-P over SPI for a wide
  bandwidth 12 kHz-class IMU, or a cheaper ADXL345 over I2C at a lower
  sampling rate — which would require retraining, since the model expects
  12 kHz input). Wiring one up is a bounded, well-defined next step: sample
  into a ring buffer, hand off `WINDOW_LENGTH`-sample windows to
  `vibration::RunInference()`.
* **On-target compile-only CI**, not flash-and-run: the CI firmware job
  (`.github/workflows/ci.yml`, job `firmware`) runs `idf.py build` inside
  Espressif's official Docker image to catch build breakage, but does not
  flash real hardware or run in an emulator — GitHub-hosted runners don't
  have an attached ESP32-S3, and esp32s3 support in QEMU is still
  experimental upstream. On-target validation (exact latency, exact arena
  usage, real-sensor accuracy) needs physical hardware.
* **OTA model updates / versioning** are not implemented; the model is
  baked into the firmware image.
