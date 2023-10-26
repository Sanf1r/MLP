#ifndef CPP7_MLP_SRC_MODEL_DATAPARSE_H_
#define CPP7_MLP_SRC_MODEL_DATAPARSE_H_

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace s21 {

class DataParse {
 public:
  DataParse() = default;

  void ParseData(const std::string &file,
                 std::vector<std::pair<int, std::vector<float>>> &data,
                 int max);
  int LineCount(const std::string &file);
};

}  // namespace s21

#endif  // CPP7_MLP_SRC_MODEL_DATAPARSE_H_
