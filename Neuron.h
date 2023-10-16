// #include <cmath>

class Neuron {
 private:
  float value_ = 0.0;
  float error_ = 0.0;
  float bias_ = 0.0;

 public:
  void setValue(float x) { value_ = x; }
  void setError(float x) { error_ = x; }
  void setBias(float x) { bias_ = x; }

  float getValue() { return value_; }
  float getError() { return error_; }
  float getBias() { return bias_; }

  void activate() { value_ = 1.0 / (1.0 + exp(-value_)); }

  float sigmoidDx() { return value_ * (1.0 - value_); }
};