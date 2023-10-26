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

#define INF 1.0 / 0.0

class gNN {
 public:
  int layersSize;
  std::vector<int> nnStruct;
  std::vector<std::vector<float>> adjMatrix_;
  int totalNeurons;
  int outputStart;
  std::vector<Neuron> neurons;
  std::vector<float> mse;
  int nnInputSize = 784;
  int nnHiddenSize = 155;
  int nnOutputSize = 26;
  int drop = 25;

  // float learn = 0.3;   mse 0.067  acc 86%
  // float learn = 0.4;   mse 0.0073 acc 83%
  // float learn = 0.35;  mse 0.0070 acc 85%
  // float learn = 0.2;   mse 0.0067 acc 85.3%
  // float learn = 0.25;  mse 0.0067 acc 85.89%
  // float learn = 0.325; mse 0.0068 acc 84.72%
  float learn = 0.3;

  explicit gNN(int input) {
    layersSize = input + 1;

    for (int i = 0; i <= layersSize; ++i) {
      if (i == 0) {
        nnStruct.push_back(nnInputSize);
      } else if (i == layersSize) {
        nnStruct.push_back(nnOutputSize);
      } else {
        nnStruct.push_back(nnHiddenSize - drop);
        drop += 25;
      }
    }

    totalNeurons = std::reduce(nnStruct.begin(), nnStruct.end());
    outputStart = totalNeurons - nnOutputSize;

    adjMatrix_.resize(totalNeurons);
    for (int i = 0; i < totalNeurons; ++i) {
      adjMatrix_[i].resize(totalNeurons, INF);
    }

    neurons.reserve(totalNeurons);
  }

  int sumStruct(int k) {
    int result = 0;
    for (int i = 0; i < k; ++i) {
      result += nnStruct[i];
    }

    return result;
  }

  int sumStructInclude(int k) {
    int result = 0;
    for (int i = 0; i <= k; ++i) {
      result += nnStruct[i];
    }

    return result;
  }

  void feedForward(const std::vector<float> &input) {
    for (int i = 0; i < nnInputSize; ++i) {
      neurons[i].setValue(input[i]);
    }

    for (int i = nnStruct[0]; i < totalNeurons; ++i) {
      float result = 0;
      for (int j = 0; j < totalNeurons; ++j) {
        if (adjMatrix_[j][i] != INF) {
          result += neurons[j].getValue() * adjMatrix_[j][i];
        }
      }
      neurons[i].setValue(result + neurons[i].getBias());
      neurons[i].activate();
    }
  }

  bool results(int ans) {
    int rightAns = ans + outputStart;
    int outIndex = 0;
    float max = 0;
    for (int i = outputStart; i < totalNeurons; ++i) {
      if (neurons[i].getValue() > max) {
        max = neurons[i].getValue();
        outIndex = i;
      }
    }
    return (rightAns == outIndex) ? true : false;
  }

  void accur(std::vector<std::pair<int, std::vector<float>>> &data) {
    int acc = 0;
    for (size_t i = 0; i < data.size(); ++i) {
      feedForward(data[i].second);
      if (results(data[i].first)) {
        ++acc;
      }
    }
    std::cout << acc << std::endl;
    std::cout << std::setprecision(4) << (float)acc / data.size() * 100 << "%"
              << std::endl;
  }
};