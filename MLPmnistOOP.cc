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
  float value = 0.0;
  float error = 0.0;
  float bias = 0.0;

 public:
  void setValue(float x) { value = x; }
  void setError(float x) { error = x; }
  void setBias(float x) { bias = x; }

  float getValue() { return value; }
  float getError() { return error; }
  float getBias() { return bias; }
};

class Layer {
 private:
  int leftNeurons;
  int rightNeurons;
  std::vector<std::vector<float>> LayerWeights;

  float genXavier(float x) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> gen(-(std::sqrt(6.0) / std::sqrt(x)),
                                              (std::sqrt(6.0) / std::sqrt(x)));
    return gen(rng);
  }

 public:
  Layer(int leftNeurons, int rightNeurons) {
    if (leftNeurons < 1 || rightNeurons < 1) {
      throw std::runtime_error("ERROR!");
    } else {
      this->leftNeurons = leftNeurons;
      this->rightNeurons = rightNeurons;
      LayerWeights.resize(leftNeurons);
      for (int i = 0; i < leftNeurons; ++i) {
        LayerWeights[i].resize(rightNeurons);
        for (int j = 0; j < rightNeurons; ++j) {
          LayerWeights[i][j] = genXavier(leftNeurons + rightNeurons);
        }
      }
    }
  }

  float getRightNeurons() { return rightNeurons; }

  float &operator()(int row, int columns) {
    if ((row >= leftNeurons || row < 0) ||
        (columns >= rightNeurons || columns < 0)) {
      throw std::out_of_range("Incorrect input, index is out of range\n");
    }
    return LayerWeights[row][columns];
  }
};

class NN {
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

  NN(int input) {
    mse.resize(88800);
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

  float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

  float sigmoidDx(float x) {
    if (fabs(x) < ((1e-9))) return 0.0;
    return x * (1.0 - x);
  }

  void feedForward(std::vector<float> &input) {
    for (int i = 0; i < nnInputSize; ++i) {
      wholeNeurons[0][i].setValue(input[i]);
    }

    for (int k = 0; k < layersSize; ++k) {
      for (int i = 0; i < nnStruct[k + 1]; ++i) {
        float result = 0;
        for (int j = 0; j < nnStruct[k]; ++j) {
          result += wholeNeurons[k][j].getValue() * weightMatrix[k](j, i);
        }
        wholeNeurons[k + 1][i].setValue(
            sigmoid(result + wholeNeurons[k + 1][i].getBias()));
      }
    }
  }

  float layerSum(std::vector<Neuron> &errors, Layer &weight, int index) {
    float result = 0;
    for (size_t j = 0; j < weight.getRightNeurons(); ++j) {
      result += weight(index, j) * errors[j].getError();
    }
    return result;
  }

  void train(int answer, std::vector<float> &input) {
    feedForward(input);

    for (int i = 0; i < nnOutputSize; ++i) {
      float t = (i == answer) ? 1.0 : 0.0;
      wholeNeurons[layersSize][i].setError(
          (t - wholeNeurons[layersSize][i].getValue()) *
          sigmoidDx(wholeNeurons[layersSize][i].getValue()));

      wholeNeurons[layersSize][i].setBias(
          wholeNeurons[layersSize][i].getBias() +
          learn * wholeNeurons[layersSize][i].getError());
    }

    for (int k = layersSize - 1; k > 0; --k) {
      for (int i = 0; i < nnStruct[k]; ++i) {
        wholeNeurons[k][i].setError(
            layerSum(wholeNeurons[k + 1], weightMatrix[k], i) *
            sigmoidDx(wholeNeurons[k][i].getValue()));
        wholeNeurons[k][i].setBias(wholeNeurons[k][i].getBias() +
                                   learn * wholeNeurons[k][i].getError());
      }
    }

    for (int k = layersSize - 1; k >= 0; --k) {
      for (int i = 0; i < nnStruct[k]; ++i) {
        float tmp = wholeNeurons[k][i].getValue() * learn;
        for (int j = 0; j < nnStruct[k + 1]; ++j) {
          weightMatrix[k](i, j) += wholeNeurons[k + 1][j].getError() * tmp;
        }
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

  float mean(int ans) {
    float result = 0.0;
    for (int i = 0; i < nnOutputSize; ++i) {
      float t = (i == ans) ? 1.0 : 0.0;
      result += std::pow(wholeNeurons[layersSize][i].getValue() - t, 2);
    }
    return result;
  }

  void printStructure() {
    for (auto data : nnStruct) std::cout << data << " ";
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

void shuffleData(std::vector<std::pair<int, std::vector<float>>> &trSet) {
  auto rng = std::default_random_engine{};
  std::shuffle(trSet.begin(), trSet.end(), rng);
}

int main() {
  NN one(2);
  std::string trainFile =
      "/Users/morfinov/Downloads/emnist-letters/emnist-letters-train.csv";
  std::string testFile =
      "/Users/morfinov/Downloads/emnist-letters/emnist-letters-test.csv";
  int lineC = lineCount(trainFile);
  int lineT = lineCount(testFile);
  int epoch = 0;
  float mseKon = 1.0 / one.nnOutputSize;
  std::cout << "NN structure - ";
  one.printStructure();
  std::cout << std::endl;
  std::vector<std::pair<int, std::vector<float>>> trSet(lineC);
  parseData(trainFile, trSet);
  std::vector<std::pair<int, std::vector<float>>> testSet(lineT);
  parseData(testFile, testSet);
  std::cout << "Start train" << std::endl;
  while (epoch < 5) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < lineC; ++i) {
      one.train(trSet[i].first, trSet[i].second);
      one.mse[i] = one.mean(trSet[i].first) * mseKon;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    shuffleData(trSet);

    std::cout << ++epoch << std::fixed << std::setprecision(4)
              << " epoch has ended " << std::endl;
    std::cout << std::fixed << std::setprecision(4) << "error - "
              << std::reduce(one.mse.begin(), one.mse.end()) / 88800
              << std::endl;
    std::cout << "time - " << duration << std::endl;
    one.accur(testSet);
  }
  std::cout << "Train end" << std::endl;

  return 0;
}