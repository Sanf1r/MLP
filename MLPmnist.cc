#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <vector>

class NN {
 public:
  int nnInputSize = 784;
  int nnHiddenSize_1 = 120;
  int nnHiddenSize_2 = 120;
  int nnOutputSize = 26;
  double learn = 0.025;

  std::vector<double> nnInputNeurons;
  std::vector<double> nnHiddenNeurons_1;
  std::vector<double> nnHiddenNeurons_2;
  std::vector<double> nnOutputNeurons;

  std::vector<double> bias_1;
  std::vector<double> bias_2;
  std::vector<double> bias_3;

  std::vector<std::vector<double>> nnInputHiddenWeight;
  std::vector<std::vector<double>> nnHiddenHiddenWeight;
  std::vector<std::vector<double>> nnHiddenOutputWeight;

  std::vector<double> mse;

  NN() {
    mse.resize(88800);
    bias_1.resize(nnHiddenSize_1);
    bias_2.resize(nnHiddenSize_2);
    bias_3.resize(nnOutputSize);
  }

  double sigmoid(double x) { return 1 / (1 + exp(-x)); }
  double sigmoidDx(double x) {
    if (fabs(x) < ((1e-9))) return 0.0;
    return x * (1.0 - x);
  }

  void feedForward(std::vector<double> &input) {
    nnInputNeurons = input;
    for (int i = 0; i < nnHiddenSize_1; ++i) {
      double result = 0;
      for (int j = 0; j < nnInputSize; ++j) {
        result += input[j] * nnInputHiddenWeight[j][i];
      }
      nnHiddenNeurons_1[i] = sigmoid(result + bias_1[i]);
    }

    for (int i = 0; i < nnHiddenSize_2; ++i) {
      double result = 0;
      for (int j = 0; j < nnHiddenSize_1; ++j) {
        result += nnHiddenNeurons_1[j] * nnHiddenHiddenWeight[j][i];
      }
      nnHiddenNeurons_2[i] = sigmoid(result + bias_2[i]);
    }

    for (int i = 0; i < nnOutputSize; ++i) {
      double result = 0;
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        result += nnHiddenNeurons_2[j] * nnHiddenOutputWeight[j][i];
      }
      nnOutputNeurons[i] = sigmoid(result + bias_3[i]);
    }
  }

  double genXavier(double x) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> gen(-(std::sqrt(6.0) / std::sqrt(x)),
                                               (std::sqrt(6.0) / std::sqrt(x)));
    return gen(rng);
  }

  double layerSum(std::vector<double> &errors,
                  std::vector<std::vector<double>> &weight, int index) {
    double result = 0;
    for (size_t j = 0; j < weight[index].size(); ++j) {
      result += weight[index][j] * errors[j];
    }
    return result;
  }

  void genWeight() {
    nnInputHiddenWeight.resize(nnInputSize);
    for (int i = 0; i < nnInputSize; ++i) {
      nnInputHiddenWeight[i].resize(nnHiddenSize_1);
      for (int j = 0; j < nnHiddenSize_1; ++j) {
        nnInputHiddenWeight[i][j] = genXavier(nnInputSize + nnHiddenSize_1);
      }
    }

    nnHiddenHiddenWeight.resize(nnHiddenSize_1);
    for (int i = 0; i < nnHiddenSize_1; ++i) {
      nnHiddenHiddenWeight[i].resize(nnHiddenSize_2);
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        nnHiddenHiddenWeight[i][j] = genXavier(nnHiddenSize_1 + nnHiddenSize_2);
      }
    }

    nnHiddenOutputWeight.resize(nnHiddenSize_2);
    for (int i = 0; i < nnHiddenSize_2; ++i) {
      nnHiddenOutputWeight[i].resize(nnOutputSize);
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] = genXavier(nnHiddenSize_2 + nnOutputSize);
      }
    }

    //   for (double &data : bias_1) data = genXavier(nnInputSize,
    //   nnHiddenSize_1); for (double &data : bias_2)
    //     data = genXavier(nnHiddenSize_1, nnHiddenSize_2);
    //   for (double &data : bias_3) data = genXavier(nnHiddenSize_2,
    //   nnOutputSize);
  }

  void train(std::vector<double> &answer, std::vector<double> &input) {
    feedForward(input);

    std::vector<double> error_1(nnHiddenSize_1);
    std::vector<double> grad_1(nnHiddenSize_1);
    std::vector<double> error_2(nnHiddenSize_2);
    std::vector<double> grad_2(nnHiddenSize_2);

    std::vector<double> error_3(nnOutputSize);
    std::vector<double> grad_3(nnOutputSize);

    for (int i = 0; i < nnOutputSize; ++i) {
      grad_3[i] = sigmoidDx(nnOutputNeurons[i]);
    }

    for (int i = 0; i < nnHiddenSize_2; ++i) {
      grad_2[i] = sigmoidDx(nnHiddenNeurons_2[i]);
    }

    for (int i = 0; i < nnHiddenSize_1; ++i) {
      grad_1[i] = sigmoidDx(nnHiddenNeurons_1[i]);
    }

    for (size_t i = 0; i < error_3.size(); ++i) {
      // error_3[i] = std::pow(nnOutputNeurons[i] - answer[i], 2) / 2;
      error_3[i] = answer[i] - nnOutputNeurons[i];
    }

    for (size_t i = 0; i < error_2.size(); ++i) {
      error_2[i] = layerSum(error_3, nnHiddenOutputWeight, i);
    }

    for (size_t i = 0; i < error_1.size(); ++i) {
      error_1[i] = layerSum(error_2, nnHiddenHiddenWeight, i);
    }

    for (int i = 0; i < nnHiddenSize_2; ++i) {
      double tmp = nnHiddenNeurons_2[i] * learn;
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] += error_3[j] * tmp * grad_3[j];
      }
    }

    for (int i = 0; i < nnHiddenSize_1; ++i) {
      double tmp = nnHiddenNeurons_1[i] * learn;
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        nnHiddenHiddenWeight[i][j] += error_2[j] * tmp * grad_2[j];
      }
    }

    for (int i = 0; i < nnInputSize; ++i) {
      double tmp = nnInputNeurons[i] * learn;
      for (int j = 0; j < nnHiddenSize_1; ++j) {
        nnInputHiddenWeight[i][j] += error_1[j] * tmp * grad_1[j];
      }
    }

    for (size_t i = 0; i < bias_3.size(); ++i) {
      bias_3[i] += learn * error_3[i] * grad_3[i];
    }
    for (size_t i = 0; i < bias_2.size(); ++i) {
      bias_2[i] += learn * error_2[i] * grad_2[i];
    }
    for (size_t i = 0; i < bias_1.size(); ++i) {
      bias_1[i] += learn * error_1[i] * grad_1[i];
    }
  }
};

void print(std::vector<double> &x) {
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << x[i] << " ";
  }
}

bool results(std::vector<double> &ans, std::vector<double> &out) {
  int ansIndex = 0;
  for (size_t i = 0; i < ans.size(); ++i) {
    if (ans[i] > 0.0) {
      ansIndex = i;
      break;
    }
  }
  double max = 0;
  int outIndex = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    if (out[i] > max) {
      max = out[i];
      outIndex = i;
    }
  }
  return (ansIndex == outIndex) ? true : false;
}

void parseData(const std::string &file, std::vector<std::vector<double>> &vect,
               std::vector<std::vector<double>> &answers) {
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
        answers[t][std::stod(temp) - 1] = 1;
        ans = false;
      } else {
        vect[t].push_back(std::stod(temp) / 255.0);
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

double mean(std::vector<double> &out, std::vector<double> &ans) {
  double result = 0;
  for (size_t i = 0; i < out.size(); ++i) {
    // result += std::pow(out[i] - ans[i], 2);
    result += std::pow(out[i] - ans[i], 2);
  }
  return result;
}

void accur(std::vector<std::vector<double>> &data,
           std::vector<std::vector<double>> &ans, NN &one) {
  int acc = 0;
  for (size_t i = 0; i < ans.size(); ++i) {
    one.feedForward(data[i]);
    if (results(ans[i], one.nnOutputNeurons)) {
      ++acc;
    }
  }
  std::cout << std::setprecision(4) << acc / (double)ans.size() * 100 << "%"
            << std::endl;
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
  int epoch = 0;
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

  // std::vector<std::vector<double>> answerset = {{1, 0}, {0, 1}, {1, 0},
  // {1, 0},
  //                                               {0, 1}, {0, 1}, {1, 0},
  //                                               {0, 1}};
  // std::vector<std::vector<double>> answerset = {{0}, {1}, {0}, {0},
  //                                               {1}, {1}, {0}, {1}};
  one.nnInputNeurons.resize(one.nnInputSize);
  one.nnHiddenNeurons_1.resize(one.nnHiddenSize_1);
  one.nnHiddenNeurons_2.resize(one.nnHiddenSize_2);
  one.nnOutputNeurons.resize(one.nnOutputSize);
  //   std::cout << one.sigmoid(1) << std::endl;
  //   std::cout << gen(rng) << std::endl;
  one.genWeight();

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

  while (epoch < 20) {
    auto t1 = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < answerset.size(); ++i) {
      one.train(answerset[i], trainset[i]);
      one.mse[i] = mean(one.nnOutputNeurons, answerset[i]) / one.nnOutputSize;
    }
    std::cout << "mse = " << one.mse[0] << std::endl;
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    // one.feedForward({0, 1, 1});
    // std::cout << one.nnOutputNeurons[0] << std::endl;
    shuffleData(trainset, answerset);
    epoch++;
    if (epoch % 1 == 0) {
      // double sum_of_elems = 0;
      // for (auto& n : one.mse) sum_of_elems += n;
      std::cout << epoch << std::fixed << std::setprecision(4)
                << " epoch has ended " << std::endl;
      std::cout << std::fixed << std::setprecision(4) << "error - "
                << std::reduce(one.mse.begin(), one.mse.end()) / 88800
                << std::endl;
      std::cout << "time - " << duration << std::endl;
      accur(testset, answersetTest, one);
    }
  }

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