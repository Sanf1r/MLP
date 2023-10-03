#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

int main() {
  // std::ifstream infile(
  //     "/Users/morfinov/Downloads/emnist-letters/emnist-letters-train.csv");
  std::ifstream infile("text.txt");
  std::string line;
  std::string temp;
  // std::vector<std::vector<int>> answers(88800, std::vector<int>(26));
  // std::vector<std::vector<double>> vect(88800);
  std::vector<std::vector<int>> answers(3, std::vector<int>(26));
  std::vector<std::vector<double>> vect(3);
  int t = 0;

  while (std::getline(infile, line)) {
    bool ans = true;
    std::istringstream ss(line);
    while (std::getline(ss, temp, ',')) {
      if (ans) {
        answers[t][std::stod(temp) - 1] = 1;
        ans = false;
      } else {
        vect[t].push_back(std::stod(temp) / 255);
      }
    }
    ++t;
    ans = 0;
  }
  for (size_t i = 0; i < vect.size(); ++i) {
    for (auto data : vect[i]) std::cout << data << " ";
    std::cout << std::endl;
  }
  for (auto read : answers) {
    for (auto data : read) {
      std::cout << data << " ";
    }
    std::cout << std::endl;
  }
}