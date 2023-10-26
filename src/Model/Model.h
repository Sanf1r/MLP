#ifndef CPP7_MLP_SRC_MODEL_MODEL_H_
#define CPP7_MLP_SRC_MODEL_MODEL_H_

#include <algorithm>
#include <cmath>
#include <execution>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stack>
#include <string>
#include <vector>

#include "DataParse.h"
#include "GraphNN.h"
#include "MatrixNN.h"

namespace s21 {

class Model {
 public:
  Model() = default;

  std::vector<float> &GetError() { return errorData_; }
  std::vector<float> &GetAccur() { return accData_; }
  std::vector<float> &GetEpoch() { return epochData_; }

  float GetAccStat() { return accuStat_; }
  float GetPrecStat() { return precStat_; }
  float GetRecStat() { return recStat_; }
  float GetMeasStat() { return measStat_; }

  std::string GetTerminalData() { return terminalData_; }
  std::string GetGuessedSymbol() { return guessedSymbol_; }

  void SetSettingsData(int layersNumber, int epochNumber, float sampleSize,
                       bool crossValidationMode, int numberOfGroup,
                       bool perceptronImplementationMatrix,
                       const std::string &fileNameTrain,
                       const std::string &fileNameTest);

  void PrepareTrain();
  void RunTrain(int epochNum);
  void PrepareCrossTrain();
  void RunCrossTrain(int epochNum);
  void PrepareGraphTrain();
  void RunGraphTrain(int epochNum);
  void PrepareGraphCrossTrain();
  void RunGraphCrossTrain(int epochNum);

  void SaveWeight(std::string &path);
  bool LoadWeight(std::string &path);
  void ButGuessClicked(const std::vector<float> &symbArray);

 private:
  std::vector<std::string> letters_ = {
      "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
      "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"};

  DataParse parser_;
  MatrixNN mNN_;
  GraphNN gNN_;

  int layersNumber_ = 0;
  int epochNumber_ = 0;
  float sampleSize_ = 0.0;
  bool crossValidationMode_ = false;
  int numberOfGroup_ = 0;
  bool perceptronImplementationMatrix_ = false;
  std::string fileNameTrain_;
  std::string fileNameTest_;

  int trainFileLines_ = 0;
  int testFileLines_ = 0;

  std::vector<float> errorData_;
  std::vector<float> accData_;
  std::vector<float> epochData_;

  float accuStat_ = 0.0;
  float precStat_ = 0.0;
  float recStat_ = 0.0;
  float measStat_ = 0.0;

  std::vector<std::pair<int, std::vector<float>>> trSet_;
  std::vector<std::pair<int, std::vector<float>>> testSet_;
  std::vector<std::vector<std::pair<int, std::vector<float>>>> crossSet_;

  std::string fileNameWeight_;
  std::string terminalData_;
  std::string guessedSymbol_;

  void Statistic();
  void CleanData();
  void FillTrainData();
  void FillTestData();
  void GatherDataPlot(int num);
  void MakeCrossSet();
  void RemakeCrossSet(int num);

  void SaveWeightMatrix(std::string &path);
  void SaveWeightGraph(std::string &path);

  void LoadWeightMatrixParse(std::vector<float> &nnStruct,
                             std::vector<std::vector<float>> &weightMatrixAll);
  void LoadWeightGraphParse(
      const std::vector<float> &nnStruct,
      const std::vector<std::vector<float>> &weightMatrixAll);
};

}  // namespace s21

#endif  // CPP7_MLP_SRC_MODEL_MODEL_H_
