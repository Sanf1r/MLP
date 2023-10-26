#ifndef CPP7_MLP_SRC_MODEL_MATRIXNN_H_
#define CPP7_MLP_SRC_MODEL_MATRIXNN_H_

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <utility>
#include <vector>

#include "Neuron.h"

namespace s21 {

class MatrixNN {
  class Layer;

 public:
  MatrixNN() = default;

  float GetError() { return msError_; }
  float GetAccur() { return accur_; }
  float GetPrecision() { return precision_; }
  float GetRecall() { return recall_; }
  float GetMeasure() { return measure_; }

  int GetNNStructSize() { return nnStruct_.size(); }

  std::vector<std::vector<float>> GetWeightMatrix(int index);
  void SetLayerWeightMatrix(
      int index, const std::vector<std::vector<float>> &layerWeights);

  void Init(int input);
  std::string PrintStructure();
  void DoTrain(std::vector<std::pair<int, std::vector<float>>> &trSet);
  void DoAccur(std::vector<std::pair<int, std::vector<float>>> &testSet);
  void DoDefault(std::vector<std::pair<int, std::vector<float>>> &trSet);
  int GuessLetter(const std::vector<float> &guessSymbolArray_);

 private:
  int layersSize_ = 0;
  std::vector<int> nnStruct_;
  std::vector<Layer> weightMatrix_;

  std::vector<std::vector<Neuron>> wholeNeurons_;
  std::vector<float> mse_;

  std::vector<std::vector<int>> confMatrix_;
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

  void CleanNN();
  float Mean(int ans);
  float Accur(std::vector<std::pair<int, std::vector<float>>> &data);
  void CleanConfMatrix();
  void Metrics();
  bool Results(int ans, std::vector<Neuron> &out);
  void Train(int answer, const std::vector<float> &input);
  void FeedForward(const std::vector<float> &input);
  void ShuffleData(std::vector<std::pair<int, std::vector<float>>> &trSet);

  class Layer {
    friend class MatrixNN;

   private:
    int leftNeurons_;
    int rightNeuron_;
    std::vector<std::vector<float>> layerWeights_;

   public:
    Layer(int leftNeurons, int rightNeurons);
    std::vector<std::vector<float>> GetWaight();
    void SetLayerWeight(const std::vector<std::vector<float>> &layerWeights);
    float &operator()(int row, int columns);
    float LayerSum(std::vector<Neuron> &errors, int index);
  };
};

}  // namespace s21

#endif  // CPP7_MLP_SRC_MODEL_MATRIXNN_H_
