#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

void shuffle(std::vector<double> &one, std::vector<double> &two) {
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::shuffle(one.begin(), one.end(), std::default_random_engine(seed));
  std::shuffle(two.begin(), two.end(), std::default_random_engine(seed));
}

int main() {
  std::vector<double> one = {1, 2, 3, 4, 5};
  std::vector<double> two = {1, 2, 3, 4, 5};
  shuffle(one, two);
  for (auto &data : one) std::cout << data << " ";
  std::cout << std::endl;
  for (auto &data : two) std::cout << data << " ";
}