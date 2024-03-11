# Implements template class so this is a header only library
add_library(array2d_lib INTERFACE)

target_include_directories(array2d_lib
    INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)