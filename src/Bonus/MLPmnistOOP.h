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

class Layer {
 private:
  int leftNeurons_;
  int rightNeuron_;
  std::vector<std::vector<float>> LayerWeights;

 public:
  Layer(int leftNeurons, int rightNeurons) {
    if (leftNeurons < 1 || rightNeurons < 1) {
      throw std::runtime_error("ERROR!");
    } else {
      leftNeurons_ = leftNeurons;
      rightNeuron_ = rightNeurons;
      std::random_device dev;
      std::mt19937 rng(dev());
      std::uniform_real_distribution<float> gen(
          -(std::sqrt(6.0) / std::sqrt(leftNeurons_ + rightNeuron_)),
          (std::sqrt(6.0) / std::sqrt(leftNeurons_ + rightNeuron_)));
      LayerWeights.resize(leftNeurons_);
      for (int i = 0; i < leftNeurons_; ++i) {
        LayerWeights[i].resize(rightNeuron_);
        for (int j = 0; j < rightNeuron_; ++j) {
          LayerWeights[i][j] = gen(rng);
        }
      }
    }
  }

  float getRightNeurons() { return rightNeuron_; }

  std::vector<std::vector<float>> &GetLayerWeights() { return LayerWeights; }

  float &operator()(int row, int columns) {
    if ((row >= leftNeurons_ || row < 0) ||
        (columns >= rightNeuron_ || columns < 0)) {
      throw std::out_of_range("Incorrect input, index is out of range\n");
    }
    return LayerWeights[row][columns];
  }
};

class mNN {
 public:
  int layersSize;
  std::vector<int> nnStruct;
  std::vector<Layer> weightMatrix;
  std::vector<std::vector<Neuron>> wholeNeurons;
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

  explicit mNN(int input) {
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
      wholeNeurons.push_back(std::vector<Neuron>(nnStruct[i]));
    }
    for (int i = 0; i < layersSize; ++i) {
      weightMatrix.push_back(Layer(nnStruct[i], nnStruct[i + 1]));
    }
  }

  void feedForward(const std::vector<float> &input) {
    for (int i = 0; i < nnInputSize; ++i) {
      wholeNeurons[0][i].setValue(input[i]);
    }

    for (int k = 0; k < layersSize; ++k) {
      for (int i = 0; i < nnStruct[k + 1]; ++i) {
        float result = 0;
        for (int j = 0; j < nnStruct[k]; ++j) {
          result += wholeNeurons[k][j].getValue() * weightMatrix[k](j, i);
        }
        wholeNeurons[k + 1][i].setValue(result +
                                        wholeNeurons[k + 1][i].getBias());
        wholeNeurons[k + 1][i].activate();
      }
    }
  }

  bool results(int ans, std::vector<Neuron> &out) {
    int outIndex = 0;
    float max = 0;
    for (int i = 0; i < nnOutputSize; ++i) {
      if (out[i].getValue() > max) {
        max = out[i].getValue();
        outIndex = i;
      }
    }
    return (ans == outIndex) ? true : false;
  }

  void accur(std::vector<std::pair<int, std::vector<float>>> &data) {
    int acc = 0;
    for (size_t i = 0; i < data.size(); ++i) {
      feedForward(data[i].second);
      if (results(data[i].first, wholeNeurons[layersSize])) {
        ++acc;
      }
    }
    std::cout << acc << std::endl;
    std::cout << std::setprecision(4) << (float)acc / data.size() * 100 << "%"
              << std::endl;
  }
};

void parseData(const std::string &file,
               std::vector<std::pair<int, std::vector<float>>> &data) {
  float norm = 1.0 / 255.0;
  std::ifstream infile(file);
  std::string line;
  std::string temp;
  int t = 0;
  std::cout << "parse start" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  while (std::getline(infile, line)) {
    bool ans = true;
    std::istringstream ss(line);
    while (std::getline(ss, temp, ',')) {
      if (ans) {
        data[t].first = std::stoi(temp) - 1;
        ans = false;
      } else {
        data[t].second.push_back(std::stof(temp) * norm);
      }
    }
    ++t;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
  std::cout << "parse end, time - " << duration << std::endl;
  infile.close();
}

int lineCount(const std::string &file) {
  int result = 0;
  std::ifstream infile(file);
  std::string line;
  while (std::getline(infile, line)) {
    ++result;
  }
  infile.close();
  return result;
}
