#include "MatrixNN.h"

namespace s21 {

void MatrixNN::CleanNN() {
  drop_ = 25;
  nnStruct_.clear();
  weightMatrix_.clear();
  for (size_t i = 0; i < wholeNeurons_.size(); ++i) {
    wholeNeurons_[i].clear();
  }
  wholeNeurons_.clear();
}

void MatrixNN::Init(int input) {
  CleanNN();
  layersSize_ = input + 1;
  for (int i = 0; i <= layersSize_; ++i) {
    if (i == 0) {
      nnStruct_.push_back(nnInputSize_);
    } else if (i == layersSize_) {
      nnStruct_.push_back(nnOutputSize_);
    } else {
      nnStruct_.push_back(nnHiddenSize_ - drop_);
      drop_ += 25;
    }
    wholeNeurons_.push_back(std::vector<Neuron>(nnStruct_[i]));
  }
  for (int i = 0; i < layersSize_; ++i) {
    weightMatrix_.push_back(Layer(nnStruct_[i], nnStruct_[i + 1]));
  }
  confMatrix_.resize(nnOutputSize_);
  for (int i = 0; i < nnOutputSize_; ++i) {
    confMatrix_[i].resize(nnOutputSize_);
  }
}

void MatrixNN::FeedForward(const std::vector<float> &input) {
  for (int i = 0; i < nnInputSize_; ++i) {
    wholeNeurons_[0][i].SetValue(input[i]);
  }

  for (int k = 0; k < layersSize_; ++k) {
    for (int i = 0; i < nnStruct_[k + 1]; ++i) {
      float result = 0;
      for (int j = 0; j < nnStruct_[k]; ++j) {
        result += wholeNeurons_[k][j].GetValue() * weightMatrix_[k](j, i);
      }
      wholeNeurons_[k + 1][i].SetValue(result +
                                       wholeNeurons_[k + 1][i].GetBias());
      wholeNeurons_[k + 1][i].Activate();
    }
  }
}

void MatrixNN::Train(int answer, const std::vector<float> &input) {
  FeedForward(input);

  for (int i = 0; i < nnOutputSize_; ++i) {
    float t = (i == answer) ? 1.0 : 0.0;
    wholeNeurons_[layersSize_][i].SetError(
        (t - wholeNeurons_[layersSize_][i].GetValue()) *
        wholeNeurons_[layersSize_][i].SigmoidDx());

    wholeNeurons_[layersSize_][i].SetBias(
        wholeNeurons_[layersSize_][i].GetBias() +
        learn_ * wholeNeurons_[layersSize_][i].GetError());
  }

  for (int k = layersSize_ - 1; k > 0; --k) {
    for (int i = 0; i < nnStruct_[k]; ++i) {
      wholeNeurons_[k][i].SetError(
          weightMatrix_[k].LayerSum(wholeNeurons_[k + 1], i) *
          wholeNeurons_[k][i].SigmoidDx());
      wholeNeurons_[k][i].SetBias(wholeNeurons_[k][i].GetBias() +
                                  learn_ * wholeNeurons_[k][i].GetError());
    }
  }

  for (int k = layersSize_ - 1; k >= 0; --k) {
    for (int i = 0; i < nnStruct_[k]; ++i) {
      float tmp = wholeNeurons_[k][i].GetValue() * learn_;
      for (int j = 0; j < nnStruct_[k + 1]; ++j) {
        weightMatrix_[k](i, j) += wholeNeurons_[k + 1][j].GetError() * tmp;
      }
    }
  }
}

void MatrixNN::ShuffleData(
    std::vector<std::pair<int, std::vector<float>>> &trSet) {
  auto rng = std::default_random_engine{};
  std::shuffle(trSet.begin(), trSet.end(), rng);
}

float MatrixNN::Mean(int ans) {
  float result = 0.0;
  for (int i = 0; i < nnOutputSize_; ++i) {
    float t = (i == ans) ? 1.0 : 0.0;
    result += std::pow(wholeNeurons_[layersSize_][i].GetValue() - t, 2);
  }
  return result;
}

float MatrixNN::Accur(std::vector<std::pair<int, std::vector<float>>> &data) {
  int acc = 0;
  CleanConfMatrix();
  for (size_t i = 0; i < data.size(); ++i) {
    FeedForward(data[i].second);
    if (Results(data[i].first, wholeNeurons_[layersSize_])) {
      ++acc;
    }
  }
  return (float)acc / data.size();
}

void MatrixNN::CleanConfMatrix() {
  for (int i = 0; i < nnOutputSize_; ++i) {
    std::fill(confMatrix_[i].begin(), confMatrix_[i].end(), 0);
  }
}

void MatrixNN::Metrics() {
  float tmpPrec = 0.0;
  float tmpRec = 0.0;
  int count = 0;
  for (int i = 0; i < nnOutputSize_; ++i) {
    float trueP = confMatrix_[i][i];
    int sumPrec = 0;
    int sumRec = 0;
    for (int j = 0; j < nnOutputSize_; ++j) {
      sumPrec += confMatrix_[j][i];
      sumRec += confMatrix_[i][j];
    }
    if (trueP != 0) {
      tmpPrec += trueP / sumPrec;
      tmpRec += trueP / sumRec;
      count++;
    }
  }

  precision_ = tmpPrec / count;
  recall_ = tmpRec / count;
  measure_ = 2 * precision_ * recall_ / (precision_ + recall_);
}

bool MatrixNN::Results(int ans, std::vector<Neuron> &out) {
  int outIndex = 0;
  float max = 0;
  for (int i = 0; i < nnOutputSize_; ++i) {
    if (out[i].GetValue() > max) {
      max = out[i].GetValue();
      outIndex = i;
    }
  }
  confMatrix_[ans][outIndex]++;
  return (ans == outIndex) ? true : false;
}

MatrixNN::Layer::Layer(int leftNeurons, int rightNeurons) {
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
    layerWeights_.resize(leftNeurons_);
    for (int i = 0; i < leftNeurons_; ++i) {
      layerWeights_[i].resize(rightNeuron_);
      for (int j = 0; j < rightNeuron_; ++j) {
        layerWeights_[i][j] = gen(rng);
      }
    }
  }
}

float &MatrixNN::Layer::operator()(int row, int columns) {
  if ((row >= leftNeurons_ || row < 0) ||
      (columns >= rightNeuron_ || columns < 0)) {
    throw std::out_of_range("Incorrect input, index is out of range\n");
  }
  return layerWeights_[row][columns];
}

float MatrixNN::Layer::LayerSum(std::vector<Neuron> &errors, int index) {
  float result = 0;
  for (int j = 0; j < rightNeuron_; ++j) {
    result += layerWeights_[index][j] * errors[j].GetError();
  }
  return result;
}

std::vector<std::vector<float>> MatrixNN::Layer::GetWaight() {
  return layerWeights_;
}

void MatrixNN::Layer::SetLayerWeight(
    const std::vector<std::vector<float>> &layerWeights) {
  layerWeights_ = layerWeights;
}

std::string MatrixNN::PrintStructure() {
  std::string result;
  for (size_t i = 0; i < nnStruct_.size(); ++i) {
    result.append(std::to_string(nnStruct_[i]));
    if (i != nnStruct_.size() - 1) result.append(" ");
  }
  return result;
}

void MatrixNN::DoTrain(std::vector<std::pair<int, std::vector<float>>> &trSet) {
  int lineC = trSet.size();
  float mseKon = 1.0 / nnOutputSize_;

  ShuffleData(trSet);
  for (int i = 0; i < lineC; ++i) {
    Train(trSet[i].first, trSet[i].second);
    mse_.push_back(Mean(trSet[i].first) * mseKon);
  }

  msError_ = std::reduce(mse_.begin(), mse_.end()) / lineC;
  mse_.clear();
}

void MatrixNN::DoAccur(
    std::vector<std::pair<int, std::vector<float>>> &testSet) {
  accur_ = Accur(testSet);
  Metrics();
}

void MatrixNN::DoDefault(
    std::vector<std::pair<int, std::vector<float>>> &trSet) {
  int acc = 0;
  CleanConfMatrix();
  int lineC = trSet.size();
  float mseKon = 1.0 / nnOutputSize_;
  for (int i = 0; i < lineC; ++i) {
    FeedForward(trSet[i].second);
    mse_.push_back(Mean(trSet[i].first) * mseKon);
    if (Results(trSet[i].first, wholeNeurons_[layersSize_])) {
      ++acc;
    }
  }
  msError_ = std::reduce(mse_.begin(), mse_.end()) / lineC;
  accur_ = (float)acc / trSet.size();
  mse_.clear();
}

int MatrixNN::GuessLetter(const std::vector<float> &guessSymbolArray_) {
  FeedForward(guessSymbolArray_);
  int outIndex = 0;
  float max = 0;
  for (int i = 0; i < nnOutputSize_; ++i) {
    if (wholeNeurons_[layersSize_][i].GetValue() > max) {
      max = wholeNeurons_[layersSize_][i].GetValue();
      outIndex = i;
    }
  }
  return outIndex;
}

std::vector<std::vector<float>> MatrixNN::GetWeightMatrix(int index) {
  return weightMatrix_.at(index).GetWaight();
}

void MatrixNN::SetLayerWeightMatrix(
    int index, const std::vector<std::vector<float>> &layerWeights) {
  weightMatrix_.at(index).SetLayerWeight(layerWeights);
}

}  // namespace s21
