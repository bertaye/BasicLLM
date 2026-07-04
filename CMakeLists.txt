cmake_minimum_required(VERSION 3.20)
project(basicllm LANGUAGES CXX)

# --- Standard ---------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to a Release build if none was specified (good for perf work).
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# --- Warnings & optimization ------------------------------------------------
# Flags differ between GCC/Clang and MSVC, so branch on the compiler.
if(MSVC)
    add_compile_options(/W4)
    add_compile_options($<$<CONFIG:Release>:/O2>)
else()
    add_compile_options(-Wall -Wextra)
    add_compile_options($<$<CONFIG:Release>:-O3>
                        $<$<CONFIG:Release>:-march=native>)
endif()

# --- Executable -------------------------------------------------------------
# Glob the sources for now; we'll list them explicitly as the project grows.
file(GLOB SOURCES CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/src/*.cpp")
file(GLOB HEADERS CONFIGURE_DEPENDS "${CMAKE_SOURCE_DIR}/header/*.h")

# Add HEADERS to the target so they show up in the Visual Studio
# Solution Explorer (they don't get compiled, just listed).
add_executable(basicllm ${SOURCES} ${HEADERS})
target_include_directories(basicllm PRIVATE "${CMAKE_SOURCE_DIR}/header")

# Mirror the on-disk folders as filters in the IDE.
source_group("Source Files" FILES ${SOURCES})
source_group("Header Files" FILES ${HEADERS})
