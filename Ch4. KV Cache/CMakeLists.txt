cmake_minimum_required(VERSION 3.20)
project(ch4_kv_cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

enable_testing()

# Absolute paths to the model files so both binaries work from any directory
set(CH4_DEFS
    MODEL_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../models/stories15M.bin"
    TOKENIZER_BIN_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../models/tokenizer.bin")

# The headers sit next to main.cpp; listed here only so IDEs show them
add_executable(ch4_tests main.cpp Tensor.h TensorOps.h Model.h Tokenizer.h BinaryFileLoader.h)
target_compile_definitions(ch4_tests PRIVATE ${CH4_DEFS})

# The story demo: run it after the tests pass
add_executable(ch4_story generate.cpp)
target_compile_definitions(ch4_story PRIVATE ${CH4_DEFS})

add_test(NAME ch4_tests COMMAND ch4_tests)
