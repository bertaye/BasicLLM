cmake_minimum_required(VERSION 3.20)
project(ch2_forward_pass LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# The headers sit next to main.cpp; listed here only so IDEs show them
add_executable(ch2_tests main.cpp Tensor.h TensorOps.h Model.h BinaryFileLoader.h)

# Absolute path to the model file so the tests work from any working directory
target_compile_definitions(ch2_tests PRIVATE
    MODEL_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../models/stories15M.bin")

add_test(NAME ch2_tests COMMAND ch2_tests)
