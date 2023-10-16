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

#include "DataParse.h"
#include "Neuron.h"

class MatrixNN {
 private:
  class Layer;

  int layersSize;
  std::vector<int> nnStruct;
  std::vector<Layer> weightMatrix;
  std::vector<std::vector<Neuron>> wholeNeurons;
  std::vector<float> mse;
  int nnInputSize = 784;
  int nnHiddenSize = 155;
  int nnOutputSize = 26;
  int drop = 25;
  float learn = 0.3;

  float mean(int ans) {
    float result = 0.0;
    for (int i = 0; i < nnOutputSize; ++i) {
      float t = (i == ans) ? 1.0 : 0.0;
      result += std::pow(wholeNeurons[layersSize][i].getValue() - t, 2);
    }
    return result;
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

  void train(int answer, std::vector<float> &input) {
    feedForward(input);

    for (int i = 0; i < nnOutputSize; ++i) {
      float t = (i == answer) ? 1.0 : 0.0;
      wholeNeurons[layersSize][i].setError(
          (t - wholeNeurons[layersSize][i].getValue()) *
          wholeNeurons[layersSize][i].sigmoidDx());

      wholeNeurons[layersSize][i].setBias(
          wholeNeurons[layersSize][i].getBias() +
          learn * wholeNeurons[layersSize][i].getError());
    }

    for (int k = layersSize - 1; k > 0; --k) {
      for (int i = 0; i < nnStruct[k]; ++i) {
        wholeNeurons[k][i].setError(
            weightMatrix[k].layerSum(wholeNeurons[k + 1], i) *
            wholeNeurons[k][i].sigmoidDx());
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
        wholeNeurons[k + 1][i].setValue(result +
                                        wholeNeurons[k + 1][i].getBias());
        wholeNeurons[k + 1][i].activate();
      }
    }
  }

  void shuffleData(std::vector<std::pair<int, std::vector<float>>> &trSet) {
    auto rng = std::default_random_engine{};
    std::shuffle(trSet.begin(), trSet.end(), rng);
  }

  class Layer {
    friend class MatrixNN;

   private:
    int leftNeurons_;
    int rightNeuron_;
    std::vector<std::vector<float>> LayerWeights_;

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
        LayerWeights_.resize(leftNeurons_);
        for (int i = 0; i < leftNeurons_; ++i) {
          LayerWeights_[i].resize(rightNeuron_);
          for (int j = 0; j < rightNeuron_; ++j) {
            LayerWeights_[i][j] = gen(rng);
          }
        }
      }
    }

    float &operator()(int row, int columns) {
      if ((row >= leftNeurons_ || row < 0) ||
          (columns >= rightNeuron_ || columns < 0)) {
        throw std::out_of_range("Incorrect input, index is out of range\n");
      }
      return LayerWeights_[row][columns];
    }

    float layerSum(std::vector<Neuron> &errors, int index) {
      float result = 0;
      for (int j = 0; j < rightNeuron_; ++j) {
        result += LayerWeights_[index][j] * errors[j].getError();
      }
      return result;
    }
  };

 public:
  MatrixNN(int input) {
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

  void printStructure() {
    for (auto data : nnStruct) std::cout << data << " ";
  }

  void doTrain(std::vector<std::pair<int, std::vector<float>>> &trSet) {
    int lineC = trSet.size();
    int epoch = 0;
    float mseKon = 1.0 / nnOutputSize;
    while (epoch < 5) {
      shuffleData(trSet);
      auto t1 = std::chrono::high_resolution_clock::now();
      for (int i = 0; i < lineC; ++i) {
        train(trSet[i].first, trSet[i].second);
        mse.push_back(mean(trSet[i].first) * mseKon);
      }
      auto t2 = std::chrono::high_resolution_clock::now();
      auto duration =
          std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

      std::cout << ++epoch << std::fixed << std::setprecision(4)
                << " epoch has ended " << std::endl;
      std::cout << std::fixed << std::setprecision(4) << "error - "
                << std::reduce(mse.begin(), mse.end()) / lineC << std::endl;
      std::cout << "time - " << duration << std::endl;
      accur(trSet);
      mse.clear();
    }
  }

  void doTest(std::vector<std::pair<int, std::vector<float>>> &testSet) {
    accur(testSet);
  }
};

int main() {
  MatrixNN one(2);
  DataParse dp;
  std::string trainFile =
      "/Users/morfinov/Downloads/emnist-letters/emnist-letters-train.csv";
  std::string testFile =
      "/Users/morfinov/Downloads/emnist-letters/emnist-letters-test.csv";
  int lineC = dp.lineCount(trainFile);
  int lineT = dp.lineCount(testFile);

  std::cout << "NN structure - ";
  one.printStructure();
  std::cout << std::endl;
  std::vector<std::pair<int, std::vector<float>>> trSet(lineC);
  dp.parseData(trainFile, trSet);
  std::vector<std::pair<int, std::vector<float>>> testSet(lineT);
  dp.parseData(testFile, testSet);
  std::cout << "Start train" << std::endl;
  one.doTrain(trSet);
  std::cout << "Train end" << std::endl;
  std::cout << "Start test" << std::endl;
  one.doTest(testSet);
  std::cout << "Test end" << std::endl;

  return 0;
}