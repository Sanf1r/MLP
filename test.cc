#include <algorithm>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

// void shuffle(std::vector<double> &one, std::vector<double> &two) {
//   unsigned seed =
//   std::chrono::system_clock::now().time_since_epoch().count();
//   std::shuffle(one.begin(), one.end(), std::default_random_engine(seed));
//   std::shuffle(two.begin(), two.end(), std::default_random_engine(seed));
// }

// void shuffleData(std::vector<float> &one) {
//   auto rng = std::default_random_engine{};
//   std::shuffle(one.begin(), one.end(), rng);
// }
// class NN {
//   class Neuron {
//    private:
//     int a = 5;
//   };
//   class Graph {
//    private:
//     int idLeft;
//     int idRight;
//     float cost;
//   }
// };

int main() {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> gen(-(std::sqrt(6.0) / std::sqrt(2)),
                                            (std::sqrt(6.0) / std::sqrt(2)));
  for (int i = 0; i < 100; ++i) std::cout << gen(rng) << std::endl;
  // int nums[] = {1, 2, 3, 4, 5};
  // int length = sizeof(nums) / sizeof(int);
  // int k = 2;
  // int index = length % k;
  // std::vector<int> test;
  // for (int i = index; i < length; ++i) {
  //   test.push_back(nums[i]);
  // }
  // for (int i = 0; i < index; ++i) {
  //   test.push_back(nums[i]);
  // }
  // for (int i = 0; i < length; ++i) {
  //   nums[i] = test[i];
  // }
  // for (auto data : nums) std::cout << data << std::endl;
  // std::vector<float> one = {1, 2, 3, 10, 4, 5};
  // std::vector<double> two = {1, 2, 3, 4, 5};
  // shuffle(one, two);
  // for (auto &data : one) std::cout << data << " ";
  // std::distance(one.begin(), std::max_element(one.begin(), one.end()));
  // std::cout << std::distance(one.begin(),
  //                            std::max_element(one.begin(), one.end()))
  //           << std::endl;
  // for (size_t i = 0; i < one.size() - 1; ++i) {
  //   for (size_t j = one.size() - 1; j > i; --j) {
  //     if (one[j - 1] > one[j]) {
  //       double tmp = one[j];
  //       one[j] = one[j - 1];
  //       one[j - 1] = tmp;
  //     }
  //   }
  // }
  // shuffleData(one);
  // for (auto &data : one) std::cout << data << " ";
  // std::cout << std::endl;
  // shuffleData(one);
  // for (auto &data : one) std::cout << data << " ";
  // std::cout << std::endl;
  // shuffleData(one);
  // for (auto &data : one) std::cout << data << " ";
  // std::cout << std::endl;
  // shuffleData(one);
  // for (auto &data : one) std::cout << data << " ";
  // std::cout << std::endl;
  // for (auto &data : two) std::cout << data << " ";
}