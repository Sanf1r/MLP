#ifndef CPP7_MLP_SRC_CONTROLLER_CONTROLLER_H_
#define CPP7_MLP_SRC_CONTROLLER_CONTROLLER_H_

#include "../Model/Model.h"

namespace s21 {

class Controller {
 public:
  explicit Controller(Model *m) : model_(m){};

  void ButtonGuessClicked(std::vector<float> &symbArray) {
    model_->ButGuessClicked(symbArray);
  }

  std::string GetTerminalData() { return model_->GetTerminalData(); }

  void SetSettingsData(int layersNumber, int epochNumber, float sampleSize,
                       bool crossValidationMode, int numberOfGroup,
                       bool perceptronImplementationMatrix,
                       const std::string &fileNameTrain,
                       const std::string &fileNameTest) {
    return model_->SetSettingsData(layersNumber, epochNumber, sampleSize,
                                   crossValidationMode, numberOfGroup,
                                   perceptronImplementationMatrix,
                                   fileNameTrain, fileNameTest);
  }

  void PrepareTrain() { return model_->PrepareTrain(); }

  void RunTrain(int epochNum) { return model_->RunTrain(epochNum); }

  void PrepareCrossTrain() { return model_->PrepareCrossTrain(); }

  void RunCrossTrain(int epochNum) { return model_->RunCrossTrain(epochNum); }

  void PrepareGraphTrain() { return model_->PrepareGraphTrain(); }

  void RunGraphTrain(int epochNum) { return model_->RunGraphTrain(epochNum); }

  void PrepareGraphCrossTrain() { return model_->PrepareGraphCrossTrain(); }

  void RunGraphCrossTrain(int epochNum) {
    return model_->RunGraphCrossTrain(epochNum);
  }

  std::vector<float> &GetError() { return model_->GetError(); }

  std::vector<float> &GetAccur() { return model_->GetAccur(); }

  std::vector<float> &getEpoch() { return model_->GetEpoch(); }

  float GetAccStat() { return model_->GetAccStat(); }
  float GetPrecStat() { return model_->GetPrecStat(); }
  float GetRecStat() { return model_->GetRecStat(); }
  float GetMeasStat() { return model_->GetMeasStat(); }

  std::string GetGuessedSymbol() { return model_->GetGuessedSymbol(); }

  void SaveWeight(std::string &path) { model_->SaveWeight(path); }

  bool LoadWeight(std::string &path) { return model_->LoadWeight(path); }

 private:
  Model *model_;
};

}  // namespace s21

#endif  //  CPP7_MLP_SRC_CONTROLLER_CONTROLLER_H_
