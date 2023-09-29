#include <cmath>
#include <iostream>
#include <random>
// #include <valarray>
#include <iomanip>
#include <vector>

class NN {
 public:
  int nnInputSize = 3;
  int nnHiddenSize_1 = 4;
  int nnHiddenSize_2 = 4;
  int nnOutputSize = 1;
  double learn = 0.5;

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
        nnHiddenOutputWeight[i][j] = genXavier(nnOutputSize, nnHiddenSize_2);
      }
    }
  }

  void train(double answer, std::vector<double> input) {
    feedForward(input);
    // std::cout << nnOutputNeurons[0] << std::endl;

    std::vector<double> error_1(nnHiddenSize_1);
    std::vector<double> grad_1(nnHiddenSize_1);
    std::vector<double> error_2(nnHiddenSize_2);
    std::vector<double> grad_2(nnHiddenSize_2);
    double error_3 = nnOutputNeurons[0] - answer;
    double grad_3 = sigmoidDx(nnOutputNeurons[0]);
    double weights_delta_3 = error_3 * grad_3;

    for (int i = 0; i < nnHiddenSize_2; ++i) {
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] -=
            nnHiddenNeurons_2[i] * weights_delta_3 * learn;
        error_1[i] = nnHiddenOutputWeight[i][j] * weights_delta_3;
      }
    }

    for (size_t i = 0; i < grad_2.size(); ++i) {
      grad_2[i] = sigmoidDx(nnHiddenNeurons_2[i]);
    }

    std::vector<double> weights_delta_2;
    for (size_t i = 0; i < grad_2.size(); ++i) {
      weights_delta_2.push_back(error_2[i] * grad_2[i]);
    }

    for (int i = 0, z = 0; i < nnHiddenSize_1; ++i) {
      for (int j = 0; j < nnHiddenSize_2; ++j) {
        nnHiddenHiddenWeight[j][i] -=
            nnHiddenNeurons_1[j] * weights_delta_2[z] * learn;
      }
      z++;
    }

    for (size_t i = 0; i < grad_1.size(); ++i) {
      grad_1[i] = sigmoidDx(nnHiddenNeurons_1[i]);
    }

    std::vector<double> weights_delta_1;
    for (size_t i = 0; i < grad_1.size(); ++i) {
      weights_delta_1.push_back(error_1[i] * grad_1[i]);
    }

    for (int i = 0, z = 0; i < nnHiddenSize_1; ++i) {
      for (int j = 0; j < nnInputSize; ++j) {
        nnInputHiddenWeight[j][i] -=
            nnInputNeurons[j] * weights_delta_1[z] * learn;
      }
      z++;
    }
  }
};

void print(std::vector<double> x) {
  for (size_t i = 0; i < x.size(); ++i) {
    std::cout << x[i] << " ";
  }
}

int main() {
  NN one;
  int epoch = 0;
  std::vector<std::vector<double>> trainset = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  };
  std::vector<double> answerset = {0, 1, 0, 0, 1, 1, 0, 1};
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

  while (epoch < 5000) {
    for (size_t i = 0; i < answerset.size(); ++i) {
      one.train(answerset[i], trainset[i]);
      one.mse[i] = pow(one.nnOutputNeurons[0] - answerset[i], 2);
      // std::cout << one.nnOutputNeurons[0] << std::endl;
    }
    // one.feedForward({0, 1, 1});
    // std::cout << one.nnOutputNeurons[0] << std::endl;
    epoch++;
    if (epoch % 100 == 0) {
      // double sum_of_elems = 0;
      // for (auto& n : one.mse) sum_of_elems += n;
      std::cout << epoch << " epoch has ended ";
      std::cout << std::fixed << std::setprecision(4) << "error - "
                << std::reduce(one.mse.begin(), one.mse.end()) / 8 << std::endl;
    }
  }

  for (int i = 0; i < 8; ++i) {
    one.feedForward(trainset[i]);
    std::cout << "Input - ";
    print(trainset[i]);
    std::cout << "expected - " << answerset[i] << " ";
    std::cout << "predict - " << one.nnOutputNeurons[0] << std::endl;
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