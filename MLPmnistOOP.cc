#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

class Neuron {
 private:
  double value = 0.0;
  double error = 0.0;
  double bias = 0.0;

 public:
  void setValue(double x) { value = x; }
  void setError(double x) { error = x; }
  void setBias(double x) { bias = x; }

  double getValue() { return value; }
  double getError() { return error; }
  double getBias() { return bias; }
};

class Layer {
 private:
  int leftNeurons;
  int rightNeurons;
  std::vector<std::vector<double>> LayerWeights;

  double genXavier(double x) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> gen(-(std::sqrt(6.0) / std::sqrt(x)),
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

  double &operator()(int row, int columns) {
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
  std::vector<double> mse;
  int nnInputSize = 784;
  int nnHiddenSize = 150;
  int nnOutputSize = 26;
  int drop = 25;

  // double learn = 0.3;   mse 0.067  acc 86%
  // double learn = 0.4;   mse 0.0073 acc 83%
  // double learn = 0.35;  mse 0.0070 acc 85%
  // double learn = 0.2;   mse 0.0067 acc 85.3%
  // double learn = 0.25;  mse 0.0067 acc 85.89%
  // double learn = 0.325; mse 0.0068 acc 84.72%
  double learn = 0.3;

  NN() {
    mse.resize(88800);
    layersSize = 4;
    for (int i = 0; i < layersSize + 2; ++i) {
      if (i == 0) {
        nnStruct.push_back(nnInputSize);
      } else if (i == layersSize + 1) {
        nnStruct.push_back(nnOutputSize);
      } else {
        nnStruct.push_back(nnHiddenSize - drop);
        drop += 25;
      }
      wholeNeurons.push_back(std::vector<Neuron>(nnStruct[i]));
    }
    for (int i = 0; i <= layersSize; ++i) {
      weightMatrix.push_back(Layer(nnStruct[i], nnStruct[i + 1]));
    }
  }

  double sigmoid(double x) { return 1.0 / (1.0 + exp(-x)); }

  double sigmoidDx(double x) {
    if (fabs(x) < ((1e-9))) return 0.0;
    return x * (1.0 - x);
  }

  void feedForward(std::vector<double> &input) {
    for (int i = 0; i < nnStruct[0]; ++i) {
      for (int j = 0; j < nnStruct[1]; ++i) {
        wholeNeurons[i][j].setValue(input[i]);
      }
    }

    for (int k = 0; k <= layersSize + 1; ++k) {
      for (int i = 0; i < nnStruct[k + 1]; ++i) {
        double result = 0;
        for (int j = 0; j < nnStruct[k]; ++j) {
          result += wholeNeurons[k][j].getValue() * weightMatrix[k](j, i);
        }
        wholeNeurons[k + 1][i].setValue(
            sigmoid(result + wholeNeurons[k + 1][i].getBias()));
      }
    }

    // for (int i = 0; i < nnHiddenSize_1; ++i) {
    //   double result = 0;
    //   for (int j = 0; j < nnInputSize; ++j) {
    //     result += nnInputNeurons[j].getValue() * nnInputHiddenWeight[j][i];
    //   }
    //   nnHiddenNeurons_1[i].setValue(
    //       sigmoid(result + nnHiddenNeurons_1[i].getBias()));
    // }

    // for (int i = 0; i < nnHiddenSize_2; ++i) {
    //   double result = 0;
    //   for (int j = 0; j < nnHiddenSize_1; ++j) {
    //     result += nnHiddenNeurons_1[j].getValue() *
    //     nnHiddenHiddenWeight[j][i];
    //   }
    //   nnHiddenNeurons_2[i].setValue(
    //       sigmoid(result + nnHiddenNeurons_2[i].getBias()));
    // }

    // for (int i = 0; i < nnOutputSize; ++i) {
    //   double result = 0;
    //   for (int j = 0; j < nnHiddenSize_2; ++j) {
    //     result += nnHiddenNeurons_2[j].getValue() *
    //     nnHiddenOutputWeight[j][i];
    //   }
    //   nnOutputNeurons[i].setValue(
    //       sigmoid(result + nnOutputNeurons[i].getBias()));
    // }
  }

  // double genXavier(double x) {
  //   std::random_device dev;
  //   std::mt19937 rng(dev());
  //   std::uniform_real_distribution<double> gen(-(std::sqrt(6.0) /
  //   std::sqrt(x)),
  //                                              (std::sqrt(6.0) /
  //                                              std::sqrt(x)));
  //   return gen(rng);
  // }

  // double layerSum(std::vector<Neuron> &errors,
  //                 std::vector<std::vector<double>> &weight, int index) {
  //   double result = 0;
  //   for (size_t j = 0; j < weight[index].size(); ++j) {
  //     result += weight[index][j] * errors[j].getError();
  //   }
  //   return result;
  // }

  // void train(std::vector<double> &answer, std::vector<double> &input) {
  //   feedForward(input);

  //   for (size_t i = 0; i < nnOutputNeurons.size(); ++i) {
  //     nnOutputNeurons[i].setError((answer[i] - nnOutputNeurons[i].getValue())
  //     *
  //                                 sigmoidDx(nnOutputNeurons[i].getValue()));
  //     nnOutputNeurons[i].setBias(nnOutputNeurons[i].getBias() +
  //                                learn * nnOutputNeurons[i].getError());
  //   }

  //   for (size_t i = 0; i < nnHiddenNeurons_2.size(); ++i) {
  //     nnHiddenNeurons_2[i].setError(
  //         layerSum(nnOutputNeurons, nnHiddenOutputWeight, i) *
  //         sigmoidDx(nnHiddenNeurons_2[i].getValue()));
  //     nnHiddenNeurons_2[i].setBias(nnHiddenNeurons_2[i].getBias() +
  //                                  learn * nnHiddenNeurons_2[i].getError());
  //   }

  //   for (size_t i = 0; i < nnHiddenNeurons_1.size(); ++i) {
  //     nnHiddenNeurons_1[i].setError(
  //         layerSum(nnHiddenNeurons_2, nnHiddenHiddenWeight, i) *
  //         sigmoidDx(nnHiddenNeurons_1[i].getValue()));
  //     nnHiddenNeurons_1[i].setError(nnHiddenNeurons_1[i].getError() +
  //                                   learn * nnHiddenNeurons_1[i].getError());
  //   }

  //   for (int i = 0; i < nnHiddenSize_2; ++i) {
  //     double tmp = nnHiddenNeurons_2[i].getValue() * learn;
  //     for (int j = 0; j < nnOutputSize; ++j) {
  //       nnHiddenOutputWeight[i][j] += nnOutputNeurons[j].getError() * tmp;
  //     }
  //   }

  //   for (int i = 0; i < nnHiddenSize_1; ++i) {
  //     double tmp = nnHiddenNeurons_1[i].getValue() * learn;
  //     for (int j = 0; j < nnHiddenSize_2; ++j) {
  //       nnHiddenHiddenWeight[i][j] += nnHiddenNeurons_2[j].getError() * tmp;
  //     }
  //   }

  //   for (int i = 0; i < nnInputSize; ++i) {
  //     double tmp = nnInputNeurons[i].getValue() * learn;
  //     for (int j = 0; j < nnHiddenSize_1; ++j) {
  //       nnInputHiddenWeight[i][j] += nnHiddenNeurons_1[j].getError() * tmp;
  //     }
  //   }
  // }

  // bool results(std::vector<double> &ans, std::vector<Neuron> &out) {
  //   size_t ansIndex =
  //       std::distance(ans.begin(), std::max_element(ans.begin(), ans.end()));
  //   size_t outIndex = 0;
  //   double max = 0;
  //   for (size_t i = 0; i < out.size(); ++i) {
  //     if (out[i].getValue() > max) {
  //       max = out[i].getValue();
  //       outIndex = i;
  //     }
  //   }
  //   return (ansIndex == outIndex) ? true : false;
  // }

  // void accur(std::vector<std::vector<double>> &data,
  //            std::vector<std::vector<double>> &ans, NN &one) {
  //   int acc = 0;
  //   for (size_t i = 0; i < ans.size(); ++i) {
  //     one.feedForward(data[i]);
  //     if (results(ans[i], nnOutputNeurons)) {
  //       ++acc;
  //     }
  //   }
  //   std::cout << std::setprecision(4) << (double)acc / ans.size() * 100 <<
  //   "%"
  //             << std::endl;
  // }

  // double mean(std::vector<double> &ans) {
  //   double result = 0.0;
  //   for (int i = 0; i < nnOutputSize; ++i) {
  //     result += std::pow(nnOutputNeurons[i].getValue() - ans[i], 2);
  //   }
  //   return result;
  // }
};

void print(std::vector<double> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << x[i] << " ";
  }
}

void parseData(const std::string &file, std::vector<std::vector<double>> &vect,
               std::vector<std::vector<double>> &answers) {
  double norm = 1.0 / 255.0;
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
        answers[t][std::stod(temp) - 1.0] = 1.0;
        ans = false;
      } else {
        vect[t].push_back(std::stod(temp) * norm);
        // vect[t].push_back(std::stod(temp) / 127.5 - 1.0);
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

void shuffleData(std::vector<std::vector<double>> &trainset,
                 std::vector<std::vector<double>> &answerset) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(trainset.begin(), trainset.end(),
               std::default_random_engine(seed));
  std::shuffle(answerset.begin(), answerset.end(),
               std::default_random_engine(seed));
}

int main() {
  NN one;
  // int epoch = 0;
  // double mseKon = 1.0 / one.nnOutputSize;
  // std::vector<std::vector<double>> trainset = {
  //     {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
  //     {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  // };
  std::vector<std::vector<double>> trainset(88800);
  std::vector<std::vector<double>> answerset(88800, std::vector<double>(26));
  parseData("/Users/morfinov/Downloads/emnist-letters/emnist-letters-train.csv",
            trainset, answerset);
  std::vector<std::vector<double>> testset(14800);
  std::vector<std::vector<double>> answersetTest(14800,
                                                 std::vector<double>(26));
  parseData("/Users/morfinov/Downloads/emnist-letters/emnist-letters-test.csv",
            testset, answersetTest);
  one.feedForward(trainset[0]);
  // std::vector<std::vector<double>> answerset = {{1, 0}, {0, 1}, {1, 0},
  // {1, 0},
  //                                               {0, 1}, {0, 1}, {1, 0},
  //                                               {0, 1}};
  // std::vector<std::vector<double>> answerset = {{0}, {1}, {0}, {0},
  //                                               {1}, {1}, {0}, {1}};
  // one.nnInputNeurons.resize(one.nnInputSize);
  // one.nnHiddenNeurons_1.resize(one.nnHiddenSize_1);
  // one.nnHiddenNeurons_2.resize(one.nnHiddenSize_2);
  // one.nnOutputNeurons.resize(one.nnOutputSize);
  //   std::cout << one.sigmoid(1) << std::endl;
  //   std::cout << gen(rng) << std::endl;
  // one.genWeight();

  // one.nnInputHiddenWeight.resize(one.nnInputSize);
  // for (size_t i = 0; i < one.nnInputHiddenWeight.size(); ++i) {
  //   one.nnInputHiddenWeight[i].resize(one.nnHiddenSize);
  // }
  // one.nnInputHiddenWeight[0][0] = 0.79;
  // one.nnInputHiddenWeight[0][1] = 0.85;
  // one.nnInputHiddenWeight[1][0] = 0.44;
  // one.nnInputHiddenWeight[1][1] = 0.43;
  // one.nnInputHiddenWeight[2][0] = 0.43;
  // one.nnInputHiddenWeight[2][1] = 0.29;

  // one.nnHiddenOutputWeight.resize(one.nnHiddenSize);
  // for (size_t i = 0; i < one.nnHiddenOutputWeight.size(); ++i) {
  //   one.nnHiddenOutputWeight[i].resize(one.nnOutputSize);
  // }
  // one.nnHiddenOutputWeight[0][0] = 0.5;
  // one.nnHiddenOutputWeight[1][0] = 0.52;

  // while (epoch < 5) {
  //   auto t1 = std::chrono::high_resolution_clock::now();
  //   for (size_t i = 0; i < answerset.size(); ++i) {
  //     one.train(answerset[i], trainset[i]);
  //     one.mse[i] = one.mean(answerset[i]) * mseKon;
  //   }
  //   std::cout << "mse = " << one.mse[0] << std::endl;
  //   auto t2 = std::chrono::high_resolution_clock::now();
  //   auto duration =
  //       std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
  //   // one.feedForward({0, 1, 1});
  //   // std::cout << one.nnOutputNeurons[0] << std::endl;
  //   shuffleData(trainset, answerset);
  //   epoch++;
  //   if (epoch % 1 == 0) {
  //     // double sum_of_elems = 0;
  //     // for (auto& n : one.mse) sum_of_elems += n;
  //     std::cout << epoch << std::fixed << std::setprecision(4)
  //               << " epoch has ended " << std::endl;
  //     std::cout << std::fixed << std::setprecision(4) << "error - "
  //               << std::reduce(one.mse.begin(), one.mse.end()) / 88800
  //               << std::endl;
  //     std::cout << "time - " << duration << std::endl;
  //     one.accur(testset, answersetTest, one);
  //     // accur(trainset, answerset, one);
  //   }
  // }

  // std::cout << one.nnOutputNeurons[0] << std::endl;
  // one.train(0, {1, 1, 0});
  // std::cout << one.nnHiddenOutputWeight[0][0] << std::endl;
  // std::cout << one.nnHiddenOutputWeight[1][0] << std::endl;

  //   if (one.nnOutputNeurons[0] > 0.5) {
  //     std::cout << "GO" << std::endl;
  //   } else {
  //     std::cout << "NOT GO" << std::endl;
  //   }
  return 0;
}