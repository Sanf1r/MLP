#include "MLPmnistOOP.h"
#include "MLPmnistOOPGraph.h"

int main() {
  mNN one(2);
  gNN two(2);

  std::string testFile = "TEST FILE PATH";
  int lineT = lineCount(testFile);
  std::vector<std::pair<int, std::vector<float>>> testSet(lineT);
  parseData(testFile, testSet);

  // parsing same weights
  std::ifstream f("WEIGHTS PATH");

  std::vector<float> nnStruct;
  std::string firstLine;
  std::string line;
  std::string temp;

  std::getline(f, firstLine);
  std::istringstream ss0(firstLine);
  while (std::getline(ss0, temp, ' ')) {
    nnStruct.push_back(std::stof(temp));
  }

  std::vector<std::vector<float>> weightMatrixAll;
  std::vector<float> weightMatrix;

  int r = 0, z = 0;
  while (std::getline(f, line)) {
    std::istringstream ss(line);
    while (std::getline(ss, temp, ' ')) {
      weightMatrix.push_back(std::stof(temp));
    }
    z++;
    if (z == nnStruct.at(r)) {
      weightMatrixAll.push_back(weightMatrix);
      weightMatrix.clear();
      r++;
      j = 0;
    }
  }

  // matrix layers fill
  std::vector<std::vector<float>> layer_1;
  layer_1.resize(784);
  int t = 0, step = 130, stop = 130;
  for (int i = 0; i < (int)weightMatrixAll[0].size(); i += step) {
    layer_1[t].insert(layer_1[t].end(), weightMatrixAll[0].begin() + i,
                      weightMatrixAll[0].begin() + stop);
    stop += 130;
    ++t;
  }

  std::vector<std::vector<float>> layer_2;
  layer_2.resize(130);
  t = 0, step = 105, stop = 105;
  for (int i = 0; i < (int)weightMatrixAll[1].size(); i += step) {
    layer_2[t].insert(layer_2[t].end(), weightMatrixAll[1].begin() + i,
                      weightMatrixAll[1].begin() + stop);
    stop += 105;
    ++t;
  }

  std::vector<std::vector<float>> layer_3;
  layer_3.resize(105);
  t = 0, step = 26, stop = 26;
  for (int i = 0; i < (int)weightMatrixAll[2].size(); i += step) {
    layer_3[t].insert(layer_3[t].end(), weightMatrixAll[2].begin() + i,
                      weightMatrixAll[2].begin() + stop);
    stop += 26;
    ++t;
  }

  one.weightMatrix[0].GetLayerWeights() = layer_1;
  one.weightMatrix[1].GetLayerWeights() = layer_2;
  one.weightMatrix[2].GetLayerWeights() = layer_3;

  // Graph matrix fill
  t = 0;
  for (int i = 0; i < 784; ++i) {
    for (int j = 784; j < 914; ++j) {
      two.adjMatrix_[i][j] = weightMatrixAll[0][t++];
    }
  }
  t = 0;
  for (int i = 784; i < 914; ++i) {
    for (int j = 914; j < 1019; ++j) {
      two.adjMatrix_[i][j] = weightMatrixAll[1][t++];
    }
  }
  t = 0;
  for (int i = 914; i < 1019; ++i) {
    for (int j = 1019; j < 1045; ++j) {
      two.adjMatrix_[i][j] = weightMatrixAll[2][t++];
    }
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    one.accur(testSet);
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto time_matrix =
      std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();

  auto t3 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; ++i) {
    two.accur(testSet);
  }
  auto t4 = std::chrono::high_resolution_clock::now();
  auto time_graph =
      std::chrono::duration_cast<std::chrono::seconds>(t4 - t3).count();

  std::cout << "matrix time = " << time_matrix << " seconds" << std::endl;

  std::cout << "graph time = " << time_graph << " seconds" << std::endl;

  return 0;
}