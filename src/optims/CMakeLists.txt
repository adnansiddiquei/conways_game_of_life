add_executable(convolutions convolutions.cpp)
ClangFormat(convolutions .)

target_link_libraries(convolutions PUBLIC array2d_lib)
target_link_libraries(convolutions PUBLIC conway_lib)