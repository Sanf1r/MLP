#ifndef CPP7_MLP_SRC_MODEL_NEURON_H_
#define CPP7_MLP_SRC_MODEL_NEURON_H_

#include <cmath>

namespace s21 {

class Neuron {
 private:
  float value_ = 0.0;
  float error_ = 0.0;
  float bias_ = 0.0;

 public:
  void SetValue(float x) { value_ = x; }
  void SetError(float x) { error_ = x; }
  void SetBias(float x) { bias_ = x; }

  float GetValue() { return value_; }
  float GetError() { return error_; }
  float GetBias() { return bias_; }

  void Activate() { value_ = 1.0 / (1.0 + exp(-value_)); }

  float SigmoidDx() { return value_ * (1.0 - value_); }
};

}  // namespace s21

#endif  // CPP7_MLP_SRC_MODEL_NEURON_H_
