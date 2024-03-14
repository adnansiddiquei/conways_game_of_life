cmake_minimum_required(VERSION 3.15...3.26)

# Set C++ standard / version
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_EXPORT_COMPILE_COMMANDS=ON)
set(CMAKE_CXX_FLAGS "-fopenmp")

if(APPLE AND EXISTS "/opt/homebrew/bin/g++-13")
    SET(CMAKE_CXX_COMPILER "/opt/homebrew/bin/g++-13")
endif()

project(
    conways_game_of_life
    VERSION 1.0
    LANGUAGES CXX
)

find_package(MPI REQUIRED)

# Add the ./cmake/ folder to CMAKE_MODULE_PATH so that we can find the custom functions defined in there
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# include the ClangFormat function so it can be used to format code in CMakeLists.txt files within src/ and tests/
include(ClangFormat)

# This will allow `ctest --test-dir build` to be run, otherwise it won't compile the tests, this must sit
# before `add_subdirectory(tests)`
enable_testing()

# this saved all of the executables to the bin/ folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Add the subdirectory containing all the source code
add_subdirectory(src)

# Add the subdirectory for your tests
add_subdirectory(tests)

