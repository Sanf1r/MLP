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

class Neuron {
 private:
  int id_ = 0;
  float value_ = 0.0;
  float error_ = 0.0;
  float bias_ = 0.0;

 public:
  Neuron() = default;
  Neuron(int x) { id_ = x; };
  void setValue(float x) { value_ = x; }
  void setError(float x) { error_ = x; }
  void setBias(float x) { bias_ = x; }

  float getValue() { return value_; }
  float getError() { return error_; }
  float getBias() { return bias_; }
  int getId() { return id_; }

  void activate() { value_ = 1.0 / (1.0 + exp(-value_)); }

  float sigmoidDx() { return value_ * (1.0 - value_); }
};

class NN {
 public:
  int layersSize;
  std::vector<int> nnStruct;
  std::vector<std::vector<float>> adjMatrix_;
  int totalNeurons;
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

  NN(int input) {
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

    adjMatrix_.resize(totalNeurons);
    for (int i = 0; i < totalNeurons; ++i) {
      adjMatrix_[i].resize(totalNeurons, INF);
    }

    // for (int i = 0; i < totalNeurons; ++i) {
    //   for (int j = 0; j < totalNeurons; ++j) {
    //     adjMatrix_[i][j] = INF;
    //   }
    // }

    for (int k = 0; k <= layersSize; ++k) {
      std::random_device dev;
      std::mt19937 rng(dev());
      std::uniform_real_distribution<float> gen(
          -(std::sqrt(6.0) / std::sqrt(nnStruct[k] + nnStruct[k + 1])),
          (std::sqrt(6.0) / std::sqrt(nnStruct[k] + nnStruct[k + 1])));
      for (int i = sumStruct(k); i < sumStruct(k + 1); ++i) {
        for (int j = sumStruct(k + 1); j < sumStructInclude(k + 1); ++j) {
          adjMatrix_[i][j] = gen(rng);
        }
      }
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

  void printOut() {
    for (int i = sumStruct(layersSize); i < sumStructInclude(layersSize); ++i) {
      std::cout << neurons[i].getValue() << " - " << i << std::endl;
    }
  }

  void feedForward(std::vector<float> &input) {
    for (int i = 0; i < nnInputSize; ++i) {
      neurons[i].setValue(input[i]);
    }

    for (int k = 0; k < layersSize; ++k) {
      for (int i = sumStruct(k + 1); i < sumStructInclude(k + 1); ++i) {
        float result = 0;
        for (int j = sumStruct(k); j < sumStruct(k + 1); ++j) {
          result += neurons[j].getValue() * adjMatrix_[j][i];
        }
        neurons[i].setValue(result + neurons[i].getBias());
        neurons[i].activate();
      }
    }
  }

  float layerSum(std::vector<std::vector<float>> &weight, int index) {
    float result = 0;
    for (int j = 0; j < totalNeurons; ++j) {
      if (weight[index][j] != INF) {
        result += weight[index][j] * neurons[j].getError();
      }
    }
    return result;
  }

  void train(int answer, std::vector<float> &input) {
    feedForward(input);

    for (int i = sumStruct(layersSize); i < nnOutputSize; ++i) {
      float t = (i == answer) ? 1.0 : 0.0;
      neurons[i].setError((t - neurons[i].getValue()) * neurons[i].sigmoidDx());
      neurons[i].setBias(neurons[i].getBias() + learn * neurons[i].getError());
    }

    for (int k = layersSize - 1; k > 0; --k) {  // MB REDO WITHOUT THIS!
      for (int i = sumStructInclude(k) - 1; i > sumStruct(k); --i) {
        neurons[i].setError(layerSum(adjMatrix_, i) * neurons[i].sigmoidDx());
        neurons[i].setBias(neurons[i].getBias() +
                           learn * neurons[i].getError());
      }
    }

    // for (int k = layersSize - 1; k >= 0; --k) {
    //   for (int i = 0; i < nnStruct[k]; ++i) {
    //     float tmp = wholeNeurons[k][i].getValue() * learn;
    //     for (int j = 0; j < nnStruct[k + 1]; ++j) {
    //       weightMatrix[k](i, j) += wholeNeurons[k + 1][j].getError() * tmp;
    //     }
    //   }
    // }

    for (int i = sumStruct(layersSize); i >= 0; --i) {
      for (int j = 0; j < totalNeurons; ++j) {
        if (adjMatrix_[i][j] != INF) {
          adjMatrix_[i][j] +=
              neurons[i].getValue() * neurons[j].getError() * learn;
        }
      }
    }
  }

  // bool results(int ans, std::vector<Neuron> &out) {
  //   int outIndex = 0;
  //   float max = 0;
  //   for (int i = 0; i < nnOutputSize; ++i) {
  //     if (out[i].getValue() > max) {
  //       max = out[i].getValue();
  //       outIndex = i;
  //     }
  //   }
  //   return (ans == outIndex) ? true : false;
  // }

  // void accur(std::vector<std::pair<int, std::vector<float>>> &data) {
  //   int acc = 0;
  //   for (size_t i = 0; i < data.size(); ++i) {
  //     feedForward(data[i].second);
  //     if (results(data[i].first, wholeNeurons[layersSize])) {
  //       ++acc;
  //     }
  //   }
  //   std::cout << acc << std::endl;
  //   std::cout << std::setprecision(4) << (float)acc / data.size() * 100
  //   <<
  //   "%"
  //             << std::endl;
  // }

  // float mean(int ans) {
  //   float result = 0.0;
  //   for (int i = 0; i < nnOutputSize; ++i) {
  //     float t = (i == ans) ? 1.0 : 0.0;
  //     result += std::pow(wholeNeurons[layersSize][i].getValue() - t, 2);
  //   }
  //   return result;
  // }

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
  // for (int i = 0; i < 1045; ++i) {
  //   if (one.adjMatrix_[914][i] != INF) c++;
  // }
  // std::cout << c << std::endl;
  // for (int i = 1045; i < 1500; ++i) {
  //   std::cout << one.adjMatrix_[914][i] << std::endl;
  // }

  // std::cout << one.adjMatrix_[914].size() << std::endl;

  // for (auto &data : one.allGraphs)
  //   std::cout << data.getLeft()->getId() << " - " <<
  //   data.getRight()->getId()
  //             << " " << std::endl;
  std::string trainFile =
      "/Users/morfinov/Downloads/emnist-letters/emnist-letters-train.csv";
  std::string testFile =
      "/Users/morfinov/Downloads/emnist-letters/emnist-letters-test.csv";
  int lineC = lineCount(trainFile);
  int lineT = lineCount(testFile);
  int epoch = 0;
  // float mseKon = 1.0 / one.nnOutputSize;
  std::cout << "NN structure - ";
  one.printStructure();
  std::cout << std::endl;
  std::vector<std::pair<int, std::vector<float>>> trSet(lineC);
  parseData(trainFile, trSet);
  std::vector<std::pair<int, std::vector<float>>> testSet(lineT);
  parseData(testFile, testSet);
  one.feedForward(trSet[0].second);
  // one.printOut();
  std::cout << "Start train" << std::endl;
  while (epoch < 1) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < lineC; ++i) {
      one.train(trSet[i].first, trSet[i].second);
      // one.mse.push_back(one.mean(trSet[i].first) * mseKon);
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
    // one.accur(testSet);
  }
  std::cout << "Train end" << std::endl;

  return 0;
}