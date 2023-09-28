#include <cmath>
#include <iostream>
#include <random>
// #include <valarray>
#include <vector>

class NN {
 public:
  int nnInputSize = 3;
  int nnHiddenSize = 2;
  int nnOutputSize = 1;
  double learn = 0.05;

  std::vector<double> nnInputNeurons;
  std::vector<double> nnHiddenNeurons;
  std::vector<double> nnOutputNeurons;

  std::vector<std::vector<double>> nnInputHiddenWeight;
  std::vector<std::vector<double>> nnHiddenOutputWeight;

  //   std::vector<double> nnHiddenBias;
  //   std::vector<double> nnOutputBias;

  double sigmoid(double x) { return 1 / (1 + exp(-x)); }
  double sigmoidDx(double x) {
    if (fabs(x) < ((1e-9))) return 0.0;
    return x * (1.0 - x);
  }

  void mult(std::vector<double> x, double y) {
    for (size_t i = 0; i < x.size(); ++i) x[i] *= y;
  }

  void feedForward(std::vector<double> input) {
    nnInputNeurons = input;
    for (int i = 0; i < nnHiddenSize; ++i) {
      double result = 0;
      for (int j = 0; j < nnInputSize; ++j) {
        result += input[j] * nnInputHiddenWeight[j][i];
      }
      nnHiddenNeurons[i] = sigmoid(result);
    }

    for (int i = 0; i < nnOutputSize; ++i) {
      double result = 0;
      for (int j = 0; j < nnHiddenSize; ++j) {
        result += nnHiddenNeurons[j] * nnHiddenOutputWeight[j][i];
      }
      nnOutputNeurons[i] = sigmoid(result);
    }
  }

  void genWeight() {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<double> gen(0.0, 1.0);

    nnInputHiddenWeight.resize(nnInputSize);
    for (int i = 0; i < nnInputSize; ++i) {
      nnInputHiddenWeight[i].resize(nnHiddenSize);
      for (int j = 0; j < nnHiddenSize; ++j) {
        nnInputHiddenWeight[i][j] = gen(rng);
      }
    }

    nnHiddenOutputWeight.resize(nnHiddenSize);
    for (int i = 0; i < nnHiddenSize; ++i) {
      nnHiddenOutputWeight[i].resize(nnOutputSize);
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] = gen(rng);
      }
    }
  }

  //   void backPropagation(double answer) {
  //     double error =
  //   }

  void train(double answer, std::vector<double> input) {
    nnInputNeurons = input;
    for (int i = 0; i < nnHiddenSize; ++i) {
      double result = 0;
      for (int j = 0; j < nnInputSize; ++j) {
        result += input[j] * nnInputHiddenWeight[j][i];
      }
      nnHiddenNeurons[i] = sigmoid(result);
    }

    for (int i = 0; i < nnOutputSize; ++i) {
      double result = 0;
      for (int j = 0; j < nnHiddenSize; ++j) {
        result += nnHiddenNeurons[j] * nnHiddenOutputWeight[j][i];
      }
      nnOutputNeurons[i] = sigmoid(result);
    }

    std::vector<double> error_1(nnHiddenSize);
    std::vector<double> grad_1(nnHiddenSize);
    double error_2 = nnOutputNeurons[0] - answer;
    double grad_2 = sigmoidDx(nnOutputNeurons[0]);
    double weights_delta_2 = error_2 * grad_2;

    for (int i = 0; i < nnHiddenSize; ++i) {
      for (int j = 0; j < nnOutputSize; ++j) {
        nnHiddenOutputWeight[i][j] -=
            nnHiddenNeurons[i] * weights_delta_2 * learn;
        error_1[i] = nnHiddenOutputWeight[i][j] * weights_delta_2;
      }
    }
    std::cout << "error_1 - " << error_1[0] << std::endl;
    std::cout << "error_1 - " << error_1[1] << std::endl;

    for (size_t i = 0; i < grad_1.size(); ++i) {
      grad_1[i] = sigmoidDx(nnHiddenNeurons[i]);
    }

    std::vector<double> weights_delta_1;
    for (size_t i = 0; i < grad_1.size(); ++i) {
      weights_delta_1.push_back(error_1[i] * grad_1[i]);
    }
    std::cout << "wd - " << weights_delta_1[0] << std::endl;
    std::cout << "wd - " << weights_delta_1[1] << std::endl;

    for (int i = 0; i < nnHiddenSize; ++i) {
      for (int j = 0; j < nnInputSize; ++j) {
        nnInputHiddenWeight[i][j] -=
            nnInputNeurons[j] * weights_delta_1[i] * learn;
      }
    }
  }
};

int main() {
  NN one;
  //   int epoch = 0;
  std::vector<std::vector<double>> trainset = {
      {0, 0, 0}, {0, 0, 1}, {0, 1, 0}, {0, 1, 1},
      {1, 0, 0}, {1, 0, 1}, {1, 1, 0}, {1, 1, 1},
  };
  std::vector<double> answerset = {0, 1, 0, 0, 1, 1, 0, 1};
  one.nnInputNeurons.resize(one.nnInputSize);
  one.nnHiddenNeurons.resize(one.nnHiddenSize);
  one.nnOutputNeurons.resize(one.nnOutputSize);
  //   std::cout << one.sigmoid(1) << std::endl;
  //   std::cout << gen(rng) << std::endl;
  //   one.genWeight();

  one.nnInputHiddenWeight.resize(one.nnInputSize);
  for (size_t i = 0; i < one.nnInputHiddenWeight.size(); ++i) {
    one.nnInputHiddenWeight[i].resize(one.nnHiddenSize);
  }
  one.nnInputHiddenWeight[0][0] = 0.79;
  one.nnInputHiddenWeight[0][1] = 0.85;
  one.nnInputHiddenWeight[1][0] = 0.44;
  one.nnInputHiddenWeight[1][1] = 0.43;
  one.nnInputHiddenWeight[2][0] = 0.43;
  one.nnInputHiddenWeight[2][1] = 0.29;

  one.nnHiddenOutputWeight.resize(one.nnHiddenSize);
  for (size_t i = 0; i < one.nnHiddenOutputWeight.size(); ++i) {
    one.nnHiddenOutputWeight[i].resize(one.nnOutputSize);
  }
  one.nnHiddenOutputWeight[0][0] = 0.5;
  one.nnHiddenOutputWeight[1][0] = 0.52;

  //   while (epoch < 30) {
  //     for (size_t i = 0; i < answerset.size(); ++i) {
  //       one.train(answerset[i], trainset[i]);
  //     }
  //     one.feedForward({0, 0, 1});
  //     std::cout << one.nnOutputNeurons[0] << std::endl;
  //     epoch++;
  // std::cout << epoch << " has ended" << std::endl;
  //   }
  one.feedForward({1, 1, 0});
  std::cout << one.nnOutputNeurons[0] << std::endl;
  one.train(0, {1, 1, 0});
  std::cout << one.nnHiddenOutputWeight[0][0] << std::endl;
  std::cout << one.nnHiddenOutputWeight[1][0] << std::endl;

  //   if (one.nnOutputNeurons[0] > 0.5) {
  //     std::cout << "GO" << std::endl;
  //   } else {
  //     std::cout << "NOT GO" << std::endl;
  //   }
  return 0;
}