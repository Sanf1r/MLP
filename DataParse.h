class DataParse {
 public:
  DataParse() = default;
  void parseData(const std::string &file,
                 std::vector<std::pair<int, std::vector<float>>> &data) {
    float norm = 1.0 / 255.0;
    std::ifstream infile(file);
    std::string line;
    std::string temp;
    int t = 0;
    std::cout << "parse start" << std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    while (std::getline(infile, line)) {
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
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count();
    std::cout << "parse end, time - " << duration << std::endl;
    infile.close();
  }

  int lineCount(const std::string &file) {
    int result = 0;
    std::ifstream infile(file);
    std::string line;
    while (std::getline(infile, line)) {
      ++result;
    }
    infile.close();
    return result;
  }
};