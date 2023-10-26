cmake_minimum_required(VERSION 3.5)

project(MLP VERSION 1.0 LANGUAGES CXX)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_compile_options(-O3 -march=native -Wall -Wextra -Werror -std=c++17 -pedantic)

find_package(Qt6 REQUIRED COMPONENTS Widgets PrintSupport)

add_executable(MLP
    main.cc
    View/View.cc
    View/View.h
    View/View.ui
    View/Paintscene.h
    View/Paintscene.cc
    View/qcustomplot_lib/qcustomplot.h
    View/qcustomplot_lib/qcustomplot.cc
    Controller/Controller.h
    Model/Model.cc
    Model/Model.h
    Model/DataParse.h
    Model/DataParse.cc
    Model/MatrixNN.h
    Model/MatrixNN.cc
    Model/Neuron.h
    Model/GraphNN.h
    Model/GraphNN.cc
)

target_link_libraries(MLP PRIVATE Qt6::Widgets Qt6::PrintSupport)
