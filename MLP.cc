#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

class NN {
 public:
  int nnInputSize = 3;
  int nnHiddenSize_1 = 5;
  int nnHiddenSize_2 = 5;
  int nnOutputSize = 2;
  double learn = 0.25;

  std::vector<double> nnInputNeurons;
  std::vector<double> nnHiddenNeurons_1;
  std::vector<double> nnHiddenNeurons_2;
  std::vector<double> nnOutputNeurons;

  std::vector<std::vector<double>> nnInputHiddenWeight;
  std::vector<std::vector<double>> nnHiddenHiddenWeight;
  std::vector<std::vector<double>> nnHiddenOutputWeight;

  std::vector<double> mse;

  NN() { mse.resize(8); }

  double sigmoid(double x) { return 1 / (1 + exp(-x)); }
  double sigmoidDx(double x) {
    if (fabs(x) < ((1e-9))) return 0.0;
    return x * (1.0 - x);
  }

  void feedForward(std::vector<double> input) {
    nnInputNeurons = input;
    for (int i = 0; i < nnHiddenSize_1; ++i) {
      double result = 0;
      for (int j = 0; j < nnInputSize; ++j) {
        result += input[j] * nnInputHiddenWeight[j][i];
      }
      nnHiddenNeurons_1[i] = sigmoid(result);
    }

    for (int i = 0; i < nnHiddenSize_2; ++i) {
      double result = 0;
      for (int j = 0; j < nnHiddenSize_1; ++j) {
        result += nnHiddenNeurons_1[j] * nnHiddenHiddenWeight[j][i];
      }
      nnHiddenNeurons_2[i] = sigmoid(result);
    }

    for (int i = 0; i < nnOutputSize; ++i) {
      double result = 0;
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        result += nnHiddenNeurons_2[j] * nnHiddenOutputWeight[j][i];
      }
      nnOutputNeurons[i] = sigmoid(result);
    }
  }

  double genXavier(double x, double y) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> gen(-(sqrt(6) / sqrt(x + y)),
                                               (sqrt(6) / sqrt(x + y)));
    return gen(rng);
  }

  double layerSum(std::vector<double> errors,
                  std::vector<std::vector<double>> weight, int index) {
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
        nnInputHiddenWeight[i][j] = genXavier(nnInputSize, nnHiddenSize_1);
      }
    }

    nnHiddenHiddenWeight.resize(nnHiddenSize_1);
    for (int i = 0; i < nnHiddenSize_1; ++i) {
      nnHiddenHiddenWeight[i].resize(nnHiddenSize_2);
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        nnHiddenHiddenWeight[i][j] = genXavier(nnHiddenSize_1, nnHiddenSize_2);
      }
    }

    nnHiddenOutputWeight.resize(nnHiddenSize_2);
    for (int i = 0; i < nnHiddenSize_2; ++i) {
      nnHiddenOutputWeight[i].resize(nnOutputSize);
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] = genXavier(nnHiddenSize_2, nnOutputSize);
      }
    }
  }

  void train(std::vector<double> answer, std::vector<double> input) {
    feedForward(input);
    // std::cout << nnOutputNeurons[0] << std::endl;

    std::vector<double> error_1(nnHiddenSize_1);
    std::vector<double> grad_1(nnHiddenSize_1);
    std::vector<double> error_2(nnHiddenSize_2);
    std::vector<double> grad_2(nnHiddenSize_2);
    // double error_3 = nnOutputNeurons[nIndex] - answer;
    // double grad_3 = sigmoidDx(nnOutputNeurons[nIndex]);
    // double weights_delta_3 = error_3 * grad_3;
    std::vector<double> error_3(nnOutputSize);
    std::vector<double> grad_3(nnOutputSize);

    for (size_t i = 0; i < grad_3.size(); ++i) {
      grad_3[i] = sigmoidDx(nnOutputNeurons[i]);
    }

    for (size_t i = 0; i < grad_2.size(); ++i) {
      grad_2[i] = sigmoidDx(nnHiddenNeurons_2[i]);
    }

    for (size_t i = 0; i < grad_1.size(); ++i) {
      grad_1[i] = sigmoidDx(nnHiddenNeurons_1[i]);
    }

    for (size_t i = 0; i < error_3.size(); ++i) {
      // error_3[i] = pow(answer[i] - nnOutputNeurons[i], 2) / 2 * grad_3[i];
      error_3[i] = nnOutputNeurons[i] - answer[i];
    }

    for (size_t i = 0; i < error_2.size(); ++i) {
      error_2[i] = layerSum(error_3, nnHiddenOutputWeight, i);
    }

    for (size_t i = 0; i < error_1.size(); ++i) {
      error_1[i] = layerSum(error_2, nnHiddenHiddenWeight, i);
    }

    for (int i = 0; i < nnHiddenSize_2; ++i) {
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] -=
            nnHiddenNeurons_2[i] * error_3[j] * grad_3[j] * learn;
      }
    }

    for (int i = 0; i < nnHiddenSize_1; ++i) {
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        nnHiddenHiddenWeight[i][j] -=
            nnHiddenNeurons_1[i] * error_2[j] * grad_2[j] * learn;
      }
    }

    for (int i = 0; i < nnInputSize; ++i) {
      for (int j = 0; j < nnHiddenSize_1; ++j) {
        nnInputHiddenWeight[i][j] -=
            nnInputNeurons[i] * error_1[j] * grad_1[j] * learn;
      }
    }
  }
};

void print(std::vector<double> x) {
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << x[i] << " ";
  }
}

bool results(std::vector<double> ans, std::vector<double> out) {
  int ansIndex;
  bool res = true;
  for (size_t i = 0; i < ans.size(); ++i) {
    if (ans[i] == 1.0) {
      ansIndex = i;
      break;
    }
  }
  if (ans[ansIndex] - out[ansIndex] > 0.2) res = false;
  return res;
}

int main() {
  NN one;
  int epoch = 0;
  std::vector<std::vector<double>> trainset = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  };
  std::vector<std::vector<double>> answerset = {{1, 0}, {0, 1}, {1, 0}, {1, 0},
                                                {0, 1}, {0, 1}, {1, 0}, {0, 1}};
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

  while (epoch < 70) {
    for (size_t i = 0; i < answerset.size(); ++i) {
      one.train(answerset[i], trainset[i]);
      one.mse[i] = ((pow(one.nnOutputNeurons[0] - answerset[i][0], 2)) +
                    pow(one.nnOutputNeurons[1] - answerset[i][1], 2)) /
                   one.nnOutputSize;
      // std::cout << one.nnOutputNeurons[0] << std::endl;
    }
    // one.feedForward({0, 1, 1});
    // std::cout << one.nnOutputNeurons[0] << std::endl;
    epoch++;
    if (epoch % 1 == 0) {
      // double sum_of_elems = 0;
      // for (auto& n : one.mse) sum_of_elems += n;
      std::cout << epoch << std::fixed << std::setprecision(4)
                << " epoch has ended " << std::endl;
      std::cout << std::fixed << std::setprecision(4) << "error - "
                << std::reduce(one.mse.begin(), one.mse.end()) / 8 << std::endl;
    }
  }

  for (size_t i = 0; i < answerset.size(); ++i) {
    one.feedForward(trainset[i]);
    // std::cout << "Input - ";
    // print(trainset[i]);
    // std::cout << "expected - ";
    // print(answerset[i]);
    // std::cout << "predict - ";
    // print(one.nnOutputNeurons);
    if (results(answerset[i], one.nnOutputNeurons)) {
      std::cout << "SUCCESS";
    } else {
      std::cout << "FAILED";
    }
    std::cout << std::endl;
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