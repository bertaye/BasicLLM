cmake_minimum_required(VERSION 3.20)
project(ch3_sampling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# Absolute paths to the model files so both binaries work from any directory
set(CH3_DEFS
    MODEL_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../models/stories15M.bin"
    TOKENIZER_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../models/tokenizer.bin")

# The headers sit next to main.cpp; listed here only so IDEs show them
add_executable(ch3_tests main.cpp Tensor.h TensorOps.h Model.h Tokenizer.h BinaryFileLoader.h)
target_compile_definitions(ch3_tests PRIVATE ${CH3_DEFS})

# The story demo: run it after the tests pass
add_executable(ch3_story generate.cpp)
target_compile_definitions(ch3_story PRIVATE ${CH3_DEFS})

add_test(NAME ch3_tests COMMAND ch3_tests)
