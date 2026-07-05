cmake_minimum_required(VERSION 3.20)
project(ch1_tensor_ops LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# Tensor.h and TensorOps.h sit next to main.cpp; listed here only so IDEs show them
add_executable(ch1_tests main.cpp Tensor.h TensorOps.h)

add_test(NAME ch1_tests COMMAND ch1_tests)
