add_library(utils_lib
    STATIC
    array2d.cpp
)

target_include_directories(utils_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)