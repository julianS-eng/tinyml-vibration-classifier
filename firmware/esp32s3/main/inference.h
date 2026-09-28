#pragma once

#include <cstdint>

namespace vibration {

enum class FaultClass : int {
  kNormal = 0,
  kImbalance = 1,
  kInnerRaceFault = 2,
  kOuterRaceFault = 3,
};

struct InferenceResult {
  FaultClass predicted_class;
  float confidence;     // dequantized softmax-equivalent score of predicted_class
  uint32_t latency_us;  // wall-clock time spent in Interpreter::Invoke()
};

// Loads the embedded model, builds the TFLite Micro interpreter and
// allocates the static tensor arena. Must be called once before
// RunInference()/RunSelfTest(). Returns false (and logs the reason) on
// failure, e.g. an unresolved op or an undersized tensor arena.
bool InitModel();

// Classifies one window of raw (float32, g units) accelerometer samples.
// `window` must have G_TEST_VECTORS_WINDOW_LENGTH samples and must already
// be on the same scale the model was trained on -- divide raw accelerometer
// readings by RAW_SIGNAL_SCALE (see model/g_vibration_model.h) before
// calling this function.
InferenceResult RunInference(const float *window);

// Runs the model against the embedded g_test_vectors (one known-label
// window per class, pre-quantized by the training pipeline) and logs
// expected vs. predicted class. Lets the model + interpreter integration be
// validated on real hardware before a sensor driver exists.
void RunSelfTest();

}  // namespace vibration
