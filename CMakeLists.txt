cmake_minimum_required(VERSION 3.15...3.26)

project(
    conways_game_of_life
    VERSION 1.0
    LANGUAGES CXX
)

# Add the ./cmake/ folder to CMAKE_MODULE_PATH so that we can find the custom functions defined in there
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# include the ClangFormat function so it can be used to format code in CMakeLists.txt files within src/ and tests/
include(ClangFormat)

# Add the subdirectory containing all the source code
add_subdirectory(src)

# Add the subdirectory for your tests
add_subdirectory(tests)

# This will allow `ctest --test-dir build` to be run, otherwise it won't compile the tests
enable_testing()

