#include "inference.h"

#include <cmath>

#include "esp_log.h"
#include "esp_timer.h"
#include "model/g_test_vectors.h"
#include "model/g_vibration_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace vibration {
namespace {

constexpr char kTag[] = "vibration_inference";
constexpr int kWindowLength = G_TEST_VECTORS_WINDOW_LENGTH;
constexpr int kNumClasses = 4;
constexpr const char *kClassNames[kNumClasses] = {"normal", "imbalance", "inner_race_fault",
                                                    "outer_race_fault"};

// Sized from this project's RAM estimate (docs/results.json ->
// cnn.tflite_int8.ram_estimate_kb, 16 KB as of the last recorded run) plus
// headroom for TFLM's own interpreter bookkeeping. If the model
// architecture changes, re-check with tflite::RecordingMicroAllocator on
// real hardware rather than guessing, and use the *measured*
// arena_used_bytes() logged by InitModel() to retune this constant.
constexpr int kTensorArenaSize = 40 * 1024;
alignas(16) uint8_t g_tensor_arena[kTensorArenaSize];

// Exactly the six ops docs/results.json -> firmware_export.required_tflite_ops
// reported for the model committed in model/g_vibration_model.cc (BatchNorm
// is fused into the preceding Conv2D by the TFLite converter, and no
// separate Quantize/Dequantize ops are needed since the model's I/O is
// already int8). Rerun `python -m tinyml_vibration.pipeline` and check that
// list again after retraining or changing the architecture -- a missing op
// fails AllocateTensors() at startup with a clear "didn't find op" log
// line, not a silent bug.
using OpResolver = tflite::MicroMutableOpResolver<6>;

const tflite::Model *g_model = nullptr;
tflite::MicroInterpreter *g_interpreter = nullptr;
TfLiteTensor *g_input = nullptr;
TfLiteTensor *g_output = nullptr;

OpResolver BuildOpResolver() {
  OpResolver resolver;
  resolver.AddConv2D();
  resolver.AddExpandDims();
  resolver.AddFullyConnected();
  resolver.AddMean();
  resolver.AddReshape();
  resolver.AddSoftmax();
  return resolver;
}

FaultClass ArgmaxClass(const TfLiteTensor *output) {
  int8_t best_value = output->data.int8[0];
  int best_index = 0;
  for (int i = 1; i < kNumClasses; ++i) {
    if (output->data.int8[i] > best_value) {
      best_value = output->data.int8[i];
      best_index = i;
    }
  }
  return static_cast<FaultClass>(best_index);
}

float DequantizedConfidence(const TfLiteTensor *output, int class_index) {
  const float scale = output->params.scale;
  const int zero_point = output->params.zero_point;
  return (output->data.int8[class_index] - zero_point) * scale;
}

}  // namespace

bool InitModel() {
  g_model = tflite::GetModel(g_vibration_model);
  if (g_model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model schema version %lu != supported %d", g_model->version(),
                TFLITE_SCHEMA_VERSION);
    return false;
  }

  static OpResolver resolver = BuildOpResolver();
  static tflite::MicroInterpreter interpreter(g_model, resolver, g_tensor_arena,
                                               kTensorArenaSize);
  g_interpreter = &interpreter;

  if (g_interpreter->AllocateTensors() != kTfLiteOk) {
    ESP_LOGE(kTag, "AllocateTensors() failed -- tensor arena too small or missing op?");
    return false;
  }

  g_input = g_interpreter->input(0);
  g_output = g_interpreter->output(0);
  ESP_LOGI(kTag, "Model ready. Arena used: %u / %d bytes",
           static_cast<unsigned>(g_interpreter->arena_used_bytes()), kTensorArenaSize);
  return true;
}

InferenceResult RunInference(const float *window) {
  const float scale = g_input->params.scale;
  const int zero_point = g_input->params.zero_point;
  for (int i = 0; i < kWindowLength; ++i) {
    int32_t quantized = static_cast<int32_t>(lroundf(window[i] / scale)) + zero_point;
    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;
    g_input->data.int8[i] = static_cast<int8_t>(quantized);
  }

  const int64_t start_us = esp_timer_get_time();
  TfLiteStatus status = g_interpreter->Invoke();
  const int64_t elapsed_us = esp_timer_get_time() - start_us;

  InferenceResult result{};
  result.latency_us = static_cast<uint32_t>(elapsed_us);
  if (status != kTfLiteOk) {
    ESP_LOGE(kTag, "Invoke() failed");
    result.predicted_class = FaultClass::kNormal;
    result.confidence = 0.0f;
    return result;
  }

  result.predicted_class = ArgmaxClass(g_output);
  result.confidence = DequantizedConfidence(g_output, static_cast<int>(result.predicted_class));
  return result;
}

void RunSelfTest() {
  for (int i = 0; i < G_TEST_VECTORS_COUNT; ++i) {
    for (int j = 0; j < kWindowLength; ++j) {
      g_input->data.int8[j] = g_test_vectors_data[i][j];
    }
    const int64_t start_us = esp_timer_get_time();
    TfLiteStatus status = g_interpreter->Invoke();
    const int64_t elapsed_us = esp_timer_get_time() - start_us;
    if (status != kTfLiteOk) {
      ESP_LOGE(kTag, "Self-test %d: Invoke() failed", i);
      continue;
    }
    FaultClass predicted = ArgmaxClass(g_output);
    float confidence = DequantizedConfidence(g_output, static_cast<int>(predicted));
    ESP_LOGI(kTag, "Self-test %d: expected=%s predicted=%s confidence=%.2f latency=%lluus", i,
             g_test_vectors_labels[i], kClassNames[static_cast<int>(predicted)], confidence,
             static_cast<unsigned long long>(elapsed_us));
  }
}

}  // namespace vibration
