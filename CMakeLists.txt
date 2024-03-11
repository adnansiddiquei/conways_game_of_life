cmake_minimum_required(VERSION 3.15...3.26)

project(
    conways_game_of_life
    VERSION 1.0
    LANGUAGES CXX
)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(ClangFormat)

add_subdirectory(src)