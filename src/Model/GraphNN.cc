#include "GraphNN.h"

namespace s21 {

void GraphNN::CleanNN() {
  drop_ = 25;
  nnStruct_.clear();
  weightMatrix_.clear();
  wholeNeurons_.clear();
}

void GraphNN::Init(int input) {
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
  }

  totalNeurons_ = std::reduce(nnStruct_.begin(), nnStruct_.end());
  outputStart_ = totalNeurons_ - nnOutputSize_;

  weightMatrix_.resize(totalNeurons_);
  for (int i = 0; i < totalNeurons_; ++i) {
    weightMatrix_[i].resize(totalNeurons_, inf);
  }

  confMatrix_.resize(nnOutputSize_);
  for (int i = 0; i < nnOutputSize_; ++i) {
    confMatrix_[i].resize(nnOutputSize_);
  }

  for (int k = 0; k < layersSize_; ++k) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_real_distribution<float> gen(
        -(std::sqrt(6.0) / std::sqrt(nnStruct_[k] + nnStruct_[k + 1])),
        (std::sqrt(6.0) / std::sqrt(nnStruct_[k] + nnStruct_[k + 1])));
    for (int i = SumStruct(k); i < SumStructInclude(k); ++i) {
      for (int j = SumStruct(k + 1); j < SumStructInclude(k + 1); ++j) {
        weightMatrix_[i][j] = gen(rng);
      }
    }
  }

  wholeNeurons_.reserve(totalNeurons_);
}

int GraphNN::SumStruct(int k) {
  int result = 0;
  for (int i = 0; i < k; ++i) {
    result += nnStruct_[i];
  }

  return result;
}

int GraphNN::SumStructInclude(int k) {
  int result = 0;
  for (int i = 0; i <= k; ++i) {
    result += nnStruct_[i];
  }

  return result;
}

void GraphNN::FeedForward(const std::vector<float> &input) {
  for (int i = 0; i < nnInputSize_; ++i) {
    wholeNeurons_[i].SetValue(input[i]);
  }

  for (int i = nnStruct_[0]; i < totalNeurons_; ++i) {
    float result = 0;
    for (int j = 0; j < outputStart_; ++j) {
      if (weightMatrix_[j][i] != inf) {
        result += wholeNeurons_[j].GetValue() * weightMatrix_[j][i];
      }
    }
    wholeNeurons_[i].SetValue(result + wholeNeurons_[i].GetBias());
    wholeNeurons_[i].Activate();
  }
}

void GraphNN::Train(int answer, const std::vector<float> &input) {
  int rightAns = outputStart_ + answer;
  FeedForward(input);

  for (int i = outputStart_; i < totalNeurons_; ++i) {
    float t = (i == rightAns) ? 1.0 : 0.0;
    wholeNeurons_[i].SetError((t - wholeNeurons_[i].GetValue()) *
                              wholeNeurons_[i].SigmoidDx());
    wholeNeurons_[i].SetBias(wholeNeurons_[i].GetBias() +
                             learn_ * wholeNeurons_[i].GetError());
  }

  for (int i = outputStart_ - 1; i >= nnStruct_[0]; --i) {
    float result = 0.0;
    for (int j = nnInputSize_; j < totalNeurons_; ++j) {
      if (weightMatrix_[i][j] != inf) {
        result += weightMatrix_[i][j] * wholeNeurons_[j].GetError();
      }
    }
    wholeNeurons_[i].SetError(result * wholeNeurons_[i].SigmoidDx());
    wholeNeurons_[i].SetBias(wholeNeurons_[i].GetBias() +
                             learn_ * wholeNeurons_[i].GetError());
  }

  for (int i = outputStart_ - 1; i >= 0; --i) {
    for (int j = nnInputSize_; j < totalNeurons_; ++j) {
      if (weightMatrix_[i][j] != inf) {
        weightMatrix_[i][j] +=
            wholeNeurons_[i].GetValue() * wholeNeurons_[j].GetError() * learn_;
      }
    }
  }
}

void GraphNN::DoDefault(
    std::vector<std::pair<int, std::vector<float>>> &trSet) {
  int acc = 0;
  int lineC = trSet.size();
  float mseKon = 1.0 / nnOutputSize_;
  for (int i = 0; i < lineC; ++i) {
    FeedForward(trSet[i].second);
    mse_.push_back(Mean(trSet[i].first) * mseKon);
    if (Results(trSet[i].first)) {
      ++acc;
    }
  }
  msError_ = std::reduce(mse_.begin(), mse_.end()) / lineC;
  accur_ = (float)acc / trSet.size();
  mse_.clear();
}

std::vector<int> GraphNN::GetNNStruct() { return nnStruct_; }

std::vector<std::vector<float>> GraphNN::GetWeightMatrix() {
  return weightMatrix_;
}

void GraphNN::SetWeightGraph(
    const std::vector<std::vector<float>> &weightMatrix) {
  for (int i = 0; i < totalNeurons_; ++i) {
    for (int j = 0; j < totalNeurons_; ++j) {
      weightMatrix_[i][j] = weightMatrix[i][j];
    }
  }
}

void GraphNN::ShuffleData(
    std::vector<std::pair<int, std::vector<float>>> &trSet) {
  auto rng = std::default_random_engine{};
  std::shuffle(trSet.begin(), trSet.end(), rng);
}

void GraphNN::DoTrain(std::vector<std::pair<int, std::vector<float>>> &trSet) {
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

void GraphNN::DoAccur(
    std::vector<std::pair<int, std::vector<float>>> &testSet) {
  accur_ = Accur(testSet);
  Metrics();
}

bool GraphNN::Results(int ans) {
  int rightAns = ans + outputStart_;
  int outIndex = 0;
  float max = 0;
  for (int i = outputStart_; i < totalNeurons_; ++i) {
    if (wholeNeurons_[i].GetValue() > max) {
      max = wholeNeurons_[i].GetValue();
      outIndex = i;
    }
  }
  confMatrix_[ans][outIndex - outputStart_]++;
  return (rightAns == outIndex) ? true : false;
}

float GraphNN::Accur(std::vector<std::pair<int, std::vector<float>>> &data) {
  int acc = 0;
  CleanConfMatrix();
  for (size_t i = 0; i < data.size(); ++i) {
    FeedForward(data[i].second);
    if (Results(data[i].first)) {
      ++acc;
    }
  }
  return (float)acc / data.size();
}

void GraphNN::CleanConfMatrix() {
  for (int i = 0; i < nnOutputSize_; ++i) {
    std::fill(confMatrix_[i].begin(), confMatrix_[i].end(), 0);
  }
}

void GraphNN::Metrics() {
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

float GraphNN::Mean(int ans) {
  int rightAns = outputStart_ + ans;
  float result = 0.0;
  for (int i = outputStart_; i < totalNeurons_; ++i) {
    float t = (i == rightAns) ? 1.0 : 0.0;
    result += std::pow(wholeNeurons_[i].GetValue() - t, 2);
  }
  return result;
}

std::string GraphNN::PrintStructure() {
  std::string result;
  for (size_t i = 0; i < nnStruct_.size(); ++i) {
    result.append(std::to_string(nnStruct_[i]));
    if (i != nnStruct_.size() - 1) result.append(" ");
  }
  return result;
}

int GraphNN::GuessLetter(const std::vector<float> &guessSymbolArray_) {
  FeedForward(guessSymbolArray_);
  int outIndex = outputStart_;
  float max = 0;
  for (int i = outputStart_; i < totalNeurons_; ++i) {
    if (wholeNeurons_[i].GetValue() > max) {
      max = wholeNeurons_[i].GetValue();
      outIndex = i;
    }
  }
  return outIndex - outputStart_;
}

}  // namespace s21
