#ifndef CPP7_MLP_SRC_MODEL_GRAPHNN_H_
#define CPP7_MLP_SRC_MODEL_GRAPHNN_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "Neuron.h"

namespace s21 {

class GraphNN {
 public:
  GraphNN() = default;

  float GetError() { return msError_; }
  float GetAccur() { return accur_; }
  float GetPrecision() { return precision_; }
  float GetRecall() { return recall_; }
  float GetMeasure() { return measure_; }

  std::vector<int> GetNNStruct();
  std::vector<std::vector<float>> GetWeightMatrix();
  void SetWeightGraph(const std::vector<std::vector<float>> &weightMatrix);

  void Init(int input);
  std::string PrintStructure();
  void DoTrain(std::vector<std::pair<int, std::vector<float>>> &trSet);
  void DoAccur(std::vector<std::pair<int, std::vector<float>>> &testSet);
  void DoDefault(std::vector<std::pair<int, std::vector<float>>> &trSet);

  int GuessLetter(const std::vector<float> &guessSymbolArray_);

 private:
  int layersSize_ = 0;
  std::vector<int> nnStruct_;
  std::vector<std::vector<float>> weightMatrix_;

  std::vector<Neuron> wholeNeurons_;
  std::vector<float> mse_;

  std::vector<std::vector<int>> confMatrix_;

  const float inf = 1.0 / 0.0;

  float msError_ = 0.0;
  float accur_ = 0.0;

  const int nnInputSize_ = 784;
  int nnHiddenSize_ = 155;
  const int nnOutputSize_ = 26;
  int drop_ = 25;
  float learn_ = 0.3;

  float precision_ = 0.0;
  float recall_ = 0.0;
  float measure_ = 0.0;

  int totalNeurons_ = 0;
  int outputStart_ = 0;

  void CleanNN();
  float Mean(int ans);
  float Accur(std::vector<std::pair<int, std::vector<float>>> &data);
  void CleanConfMatrix();
  void Metrics();
  bool Results(int ans);
  void Train(int answer, const std::vector<float> &input);
  void FeedForward(const std::vector<float> &input);
  void ShuffleData(std::vector<std::pair<int, std::vector<float>>> &trSet);
  int SumStruct(int k);
  int SumStructInclude(int k);
};

}  // namespace s21

#endif  // CPP7_MLP_SRC_MODEL_GRAPHNN_H_
