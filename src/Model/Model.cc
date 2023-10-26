#include "Model.h"

namespace s21 {

/**
 * @brief Sets the values of class variables necessary for start neuron training
 *
 * @param layersNumber
 * @param epochNumber
 * @param sampleSize
 * @param crossValidationMode
 * @param numberOfGroup
 * @param perceptronImplementationMatrix
 * @param fileNameTrain
 * @param fileNameTest
 */
void Model::SetSettingsData(int layersNumber, int epochNumber, float sampleSize,
                            bool crossValidationMode, int numberOfGroup,
                            bool perceptronImplementationMatrix,
                            const std::string &fileNameTrain,
                            const std::string &fileNameTest) {
  layersNumber_ = layersNumber;
  epochNumber_ = epochNumber;
  sampleSize_ = sampleSize;
  crossValidationMode_ = crossValidationMode;
  numberOfGroup_ = numberOfGroup;
  perceptronImplementationMatrix_ = perceptronImplementationMatrix;
  fileNameTrain_ = fileNameTrain;
  fileNameTest_ = fileNameTest;
}

/**
 * @brief Prepares infrastructure for training a neural network as matrix
 *
 */
void Model::PrepareTrain() {
  mNN_.Init(layersNumber_);
  CleanData();
  FillTrainData();
  FillTestData();
  mNN_.DoDefault(trSet_);
  GatherDataPlot(0);
  terminalData_.append("Neural Network structure - " + mNN_.PrintStructure() +
                       ".\n");
  terminalData_.append("Matrix implementation.\n");
  terminalData_.append("Start train.\n");
}

/**
 * @brief Start training a neural network as matrix
 *
 * @param epochNum
 */
void Model::RunTrain(int epochNum) {
  terminalData_.append("Epoch " + std::to_string(epochNum) + " started.\n");
  auto t3 = std::chrono::high_resolution_clock::now();
  mNN_.DoTrain(trSet_);
  auto t4 = std::chrono::high_resolution_clock::now();
  mNN_.DoAccur(testSet_);
  Statistic();
  terminalData_.append(
      "Epoch " + std::to_string(epochNum) + " stats:\n" +
      "mean square error = " + std::to_string(mNN_.GetError()) +
      "\ntest set accuracy = " + std::to_string(mNN_.GetAccur()) + "\n");
  auto single =
      std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
  terminalData_.append("time = " + std::to_string(single) + " seconds.\n");
  terminalData_.append("\n");
  GatherDataPlot(epochNum);

  if (epochNum == epochNumber_) {
    trSet_.clear();
    testSet_.clear();
  }
}

/**
 * @brief Prepares infrastructure for training a neural network as Cross-Train
 *
 */
void Model::PrepareCrossTrain() {
  mNN_.Init(layersNumber_);
  CleanData();
  FillTrainData();
  MakeCrossSet();

  terminalData_.append("Neural Network structure - " + mNN_.PrintStructure() +
                       ".\n");
  terminalData_.append("Matrix implementation.\n");
  terminalData_.append("Start cross-validation training.\n");
}

/**
 * @brief Start training a neural network as Cross-Train
 *
 * @param epochNum
 */
void Model::RunCrossTrain(int epochNum) {
  terminalData_.append("Epoch " + std::to_string(epochNum) + " started.\n");
  RemakeCrossSet(epochNum);
  auto t3 = std::chrono::high_resolution_clock::now();
  mNN_.DoTrain(trSet_);
  auto t4 = std::chrono::high_resolution_clock::now();
  mNN_.DoAccur(crossSet_[epochNum - 1]);
  Statistic();

  terminalData_.append(
      "Epoch " + std::to_string(epochNum) + " stats:\n" +
      "mean square error = " + std::to_string(mNN_.GetError()) +
      "\ntest set accuracy = " + std::to_string(mNN_.GetAccur()) + "\n");
  auto single =
      std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
  terminalData_.append("time = " + std::to_string(single) + " seconds.\n");
  terminalData_.append("\n");
  GatherDataPlot(epochNum);

  if (epochNum == epochNumber_) {
    trSet_.clear();
    testSet_.clear();
    crossSet_.clear();
  }
}

/**
 * @brief Prepares infrastructure for training a neural network as graph
 *
 */
void Model::PrepareGraphTrain() {
  gNN_.Init(layersNumber_);
  CleanData();
  FillTrainData();
  FillTestData();
  gNN_.DoDefault(trSet_);
  GatherDataPlot(0);
  terminalData_.append("Neural Network structure - " + gNN_.PrintStructure() +
                       ".\n");
  terminalData_.append("Graph implementation.\n");
  terminalData_.append("Start train.\n");
}

/**
 * @brief Start training a neural network as graph
 *
 * @param epochNum
 */
void Model::RunGraphTrain(int epochNum) {
  terminalData_.append("Epoch " + std::to_string(epochNum) + " started.\n");
  auto t3 = std::chrono::high_resolution_clock::now();
  gNN_.DoTrain(trSet_);
  auto t4 = std::chrono::high_resolution_clock::now();
  gNN_.DoAccur(testSet_);
  Statistic();

  terminalData_.append(
      "Epoch " + std::to_string(epochNum) + " stats:\n" +
      "mean square error = " + std::to_string(gNN_.GetError()) +
      "\ntest set accuracy = " + std::to_string(gNN_.GetAccur()) + "\n");
  auto single =
      std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
  terminalData_.append("time = " + std::to_string(single) + " seconds.\n");
  terminalData_.append("\n");
  GatherDataPlot(epochNum);

  if (epochNum == epochNumber_) {
    trSet_.clear();
    testSet_.clear();
  }
}

/**
 * @brief Prepares infrastructure for training a neural network as Cross-Graph
 *
 */
void Model::PrepareGraphCrossTrain() {
  gNN_.Init(layersNumber_);
  CleanData();
  FillTrainData();
  MakeCrossSet();

  terminalData_.append("Neural Network structure - " + gNN_.PrintStructure() +
                       ".\n");
  terminalData_.append("Graph implementation.\n");
  terminalData_.append("Start cross-validation training.\n");
}

/**
 * @brief Start training a neural network as Cross-Graph
 *
 * @param epochNum
 */
void Model::RunGraphCrossTrain(int epochNum) {
  terminalData_.append("Epoch " + std::to_string(epochNum) + " started.\n");
  RemakeCrossSet(epochNum);
  auto t3 = std::chrono::high_resolution_clock::now();
  gNN_.DoTrain(trSet_);
  auto t4 = std::chrono::high_resolution_clock::now();
  gNN_.DoAccur(crossSet_[epochNum - 1]);
  Statistic();

  terminalData_.append(
      "Epoch " + std::to_string(epochNum) + " stats:\n" +
      "mean square error = " + std::to_string(gNN_.GetError()) +
      "\ntest set accuracy = " + std::to_string(gNN_.GetAccur()) + "\n");
  auto single =
      std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();
  terminalData_.append("time = " + std::to_string(single) + " seconds.\n");
  terminalData_.append("\n");
  GatherDataPlot(epochNum);

  if (epochNum == epochNumber_) {
    trSet_.clear();
    testSet_.clear();
    crossSet_.clear();
  }
}

/**
 * @brief Clean the infrastructure data
 *
 */
void Model::CleanData() {
  terminalData_.clear();
  errorData_.clear();
  accData_.clear();
  epochData_.clear();
}

/**
 * @brief Fill out the array of data from the train file
 *
 */
void Model::FillTrainData() {
  trainFileLines_ = parser_.LineCount(fileNameTrain_);
  trSet_.resize(trainFileLines_);
  parser_.ParseData(fileNameTrain_, trSet_, trainFileLines_);
}

/**
 * @brief Fill out the array of data from the test file
 *
 */
void Model::FillTestData() {
  testFileLines_ = parser_.LineCount(fileNameTest_);
  testSet_.resize(testFileLines_ * sampleSize_);
  parser_.ParseData(fileNameTest_, testSet_, testFileLines_ * sampleSize_);
}

/**
 * @brief Collect data depending on the type of neural network
 *
 * @param num
 */
void Model::GatherDataPlot(int num) {
  if (perceptronImplementationMatrix_) {
    errorData_.push_back(mNN_.GetError());
    accData_.push_back(mNN_.GetAccur());
    epochData_.push_back(num);
  } else {
    errorData_.push_back(gNN_.GetError());
    accData_.push_back(gNN_.GetAccur());
    epochData_.push_back(num);
  }
}

/**
 * @brief Creates an array for training in Cross mode
 *
 */
void Model::MakeCrossSet() {
  crossSet_.resize(numberOfGroup_);
  int koef = trainFileLines_ / numberOfGroup_;
  int t = 0;
  for (int i = 0; i < trainFileLines_; ++i) {
    if (i == koef) {
      ++t;
      koef += koef;
    }
    crossSet_[t].push_back(trSet_[i]);
  }
}

/**
 * @brief Remake the array for training in Cross mode
 *
 * @param num
 */
void Model::RemakeCrossSet(int num) {
  trSet_.clear();
  for (int i = 0; i < numberOfGroup_; ++i) {
    if (i == num - 1) continue;
    trSet_.insert(trSet_.end(), crossSet_[i].begin(), crossSet_[i].end());
  }
}

/**
 * @brief
 *
 */
void Model::Statistic() {
  accuStat_ =
      (perceptronImplementationMatrix_) ? mNN_.GetAccur() : gNN_.GetAccur();
  precStat_ = (perceptronImplementationMatrix_) ? mNN_.GetPrecision()
                                                : gNN_.GetPrecision();
  recStat_ =
      (perceptronImplementationMatrix_) ? mNN_.GetRecall() : gNN_.GetRecall();
  measStat_ =
      (perceptronImplementationMatrix_) ? mNN_.GetMeasure() : gNN_.GetMeasure();
}

/**
 * @brief Save weights depending of the neural network mode
 *
 * @param path
 */
void Model::SaveWeight(std::string &path) {
  if (perceptronImplementationMatrix_) {
    SaveWeightMatrix(path);
  } else {
    SaveWeightGraph(path);
  }
}

/**
 * @brief Save weights depending of the neural network mode as MATRIX
 *
 * @param path
 */
void Model::SaveWeightMatrix(std::string &path) {
  std::string nnStructure = mNN_.PrintStructure();
  std::vector<std::vector<float>> GetWeightMatrix;
  int weightCount = mNN_.GetNNStructSize() - 1;

  std::ofstream f(path);
  f << "Matrix" << '\n';
  f << nnStructure << '\n';
  for (int i = 0; i < weightCount; ++i) {
    GetWeightMatrix = mNN_.GetWeightMatrix(i);
    for (size_t j = 0; j < GetWeightMatrix.size(); ++j) {
      for (std::vector<float>::const_iterator k = GetWeightMatrix.at(j).begin();
           k != GetWeightMatrix.at(j).end(); ++k) {
        f << *k << ' ';
      }
      f << '\n';
    }
  }
  f.close();
}

/**
 * @brief Save weights depending of the neural network mode as GRAPH
 *
 * @param path
 */
void Model::SaveWeightGraph(std::string &path) {
  std::string nnStructure = gNN_.PrintStructure();
  std::vector<std::vector<float>> GetWeightMatrix;

  std::ofstream f(path);
  f << "Graph" << '\n';
  f << nnStructure << '\n';

  GetWeightMatrix = gNN_.GetWeightMatrix();
  for (size_t i = 0; i < GetWeightMatrix.size(); ++i) {
    for (std::vector<float>::const_iterator j = GetWeightMatrix.at(i).begin();
         j != GetWeightMatrix.at(i).end(); ++j) {
      f << *j << ' ';
    }
    f << '\n';
  }

  f.close();
}

/**
 * @brief Load weights depending of the neural network mode
 *
 * @param path
 * @return true
 * @return false
 */
bool Model::LoadWeight(std::string &path) {
  std::string firstLine;
  std::ifstream f(path);

  terminalData_.clear();
  terminalData_.append("Start loading new weights...\n");

  std::getline(f, firstLine);
  if (firstLine == "Matrix") {
    perceptronImplementationMatrix_ = true;
  } else if (firstLine == "Graph") {
    perceptronImplementationMatrix_ = false;
  } else {
    terminalData_.append(
        "Wrong file format!\nPlease check file and try again...");
    return perceptronImplementationMatrix_;
  }

  std::vector<float> nnStruct;
  std::string line;
  std::string temp;

  std::getline(f, firstLine);
  std::istringstream ss0(firstLine);
  while (std::getline(ss0, temp, ' ')) {
    nnStruct.push_back(std::stof(temp));
  }

  if (nnStruct.size() == 0) {
    terminalData_.append(
        "Wrong file format!\nPlease check file and try again...");
    return perceptronImplementationMatrix_;
  }

  std::vector<std::vector<float>> weightMatrixAll;
  std::vector<float> weightMatrix;

  if (perceptronImplementationMatrix_) {
    int i = 0, j = 0;
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      while (std::getline(ss, temp, ' ')) {
        weightMatrix.push_back(std::stof(temp));
      }
      j++;
      if (j == nnStruct.at(i)) {
        weightMatrixAll.push_back(weightMatrix);
        weightMatrix.clear();
        i++;
        j = 0;
      }
    }
  } else {
    while (std::getline(f, line)) {
      std::istringstream ss(line);
      while (std::getline(ss, temp, ' ')) {
        weightMatrix.push_back(std::stof(temp));
      }
      weightMatrixAll.push_back(weightMatrix);
      weightMatrix.clear();
    }
  }

  f.close();

  if (perceptronImplementationMatrix_) {
    LoadWeightMatrixParse(nnStruct, weightMatrixAll);
  } else {
    LoadWeightGraphParse(nnStruct, weightMatrixAll);
  }

  return perceptronImplementationMatrix_;
}

/**
 * @brief Load weights depending of the neural network mode as MATRIX
 *
 * @param nnStruct
 * @param weightMatrixAll
 */
void Model::LoadWeightMatrixParse(
    std::vector<float> &nnStruct,
    std::vector<std::vector<float>> &weightMatrixAll) {
  layersNumber_ = nnStruct.size() - 2;

  mNN_.Init(layersNumber_);
  errorData_.clear();
  accData_.clear();
  epochData_.clear();

  for (size_t i = 0; i < nnStruct.size() - 1; ++i) {
    std::vector<float> tmp(weightMatrixAll.at(i).begin(),
                           weightMatrixAll.at(i).end());
    std::vector<std::vector<float>> newTmp;
    newTmp.resize(nnStruct.at(i));
    for (size_t k = 0, count = 0; k < nnStruct.at(i); ++k) {
      newTmp.at(k).resize((nnStruct.at(i + 1)));
      for (size_t j = 0; j < nnStruct.at(i + 1); ++j, ++count) {
        newTmp.at(k).at(j) = tmp.at(count);
      }
    }
    mNN_.SetLayerWeightMatrix(i, newTmp);
  }

  terminalData_.append("End of loading new weights...\n");
  terminalData_.append("New Neural Network structure - " +
                       mNN_.PrintStructure() + ".\n");
}

/**
 * @brief  Load weights depending of the neural network mode as GRAPH
 *
 * @param nnStruct
 * @param weightMatrixAll
 */
void Model::LoadWeightGraphParse(
    const std::vector<float> &nnStruct,
    const std::vector<std::vector<float>> &weightMatrixAll) {
  layersNumber_ = nnStruct.size() - 2;
  gNN_.Init(layersNumber_);
  errorData_.clear();
  accData_.clear();
  epochData_.clear();

  gNN_.SetWeightGraph(weightMatrixAll);

  terminalData_.append("End of loading new weights...\n");
  terminalData_.append("New Neural Network structure - " +
                       gNN_.PrintStructure() + ".\n");
}

/**
 * @brief Symbol recognition depending on the regime of the neural network
 *
 * @param symbArray
 */
void Model::ButGuessClicked(const std::vector<float> &symbArray) {
  if (layersNumber_ > 0)
    guessedSymbol_ = (perceptronImplementationMatrix_)
                         ? letters_.at(mNN_.GuessLetter(symbArray))
                         : letters_.at(gNN_.GuessLetter(symbArray));
}

}  // namespace s21
