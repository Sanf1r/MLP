#include "DataParse.h"

namespace s21 {

void DataParse::ParseData(const std::string &file,
                          std::vector<std::pair<int, std::vector<float>>> &data,
                          int max) {
  float norm = 1.0 / 255.0;
  std::ifstream infile(file);
  std::string line;
  std::string temp;
  int t = 0;
  int count = 0;
  while (count < max && std::getline(infile, line)) {
    bool ans = true;
    std::istringstream ss(line);
    while (std::getline(ss, temp, ',')) {
      if (ans) {
        data[t].first = std::stoi(temp) - 1;
        ans = false;
      } else {
        data[t].second.push_back(std::stof(temp) * norm);
      }
    }
    ++t;
    ++count;
  }
  infile.close();
}

int DataParse::LineCount(const std::string &file) {
  int result = 0;
  std::ifstream infile(file);
  std::string line;
  while (std::getline(infile, line)) {
    ++result;
  }
  infile.close();
  return result;
}

}  // namespace s21
