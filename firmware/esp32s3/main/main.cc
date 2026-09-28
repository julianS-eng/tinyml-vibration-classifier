#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "inference.h"

namespace {
constexpr char kTag[] = "main";
}

extern "C" void app_main(void) {
  if (!vibration::InitModel()) {
    ESP_LOGE(kTag, "Model initialization failed; halting.");
    return;
  }

  ESP_LOGI(kTag, "Running embedded self-test (no sensor required)...");
  vibration::RunSelfTest();

  // TODO(sensor integration, tracked as future work -- see
  // firmware/esp32s3/README.md "Future work"): replace this stub with a
  // real accelerometer driver (e.g. ICM-42688-P over SPI, or ADXL345 over
  // I2C), sampled at the 12 kHz the model was trained on, buffering
  // WINDOW_LENGTH samples before each vibration::RunInference() call.
  ESP_LOGW(kTag, "No accelerometer driver wired up yet -- see README 'Future work'.");
  while (true) {
    vTaskDelay(pdMS_TO_TICKS(5000));
  }
}
