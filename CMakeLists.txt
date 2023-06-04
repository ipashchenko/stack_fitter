cmake_minimum_required(VERSION 3.24)
project(stack_fitter)

include_directories("/usr/include/eigen3", "/home/ilya/github/DNest4/code" "src")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -pthread -std=c++11 -march=native  -DNDEBUG -fopenmp")

add_executable(stack_fitter src/main.cpp src/utils.h src/utils.cpp src/kernels.h src/kernels.cpp src/Data.h src/Data.cpp src/ModelGP.cpp src/ModelGPLatent.h src/ModelGPLatent.cpp)

TARGET_LINK_LIBRARIES(stack_fitter /home/ilya/github/DNest4/code/libdnest4.a)
