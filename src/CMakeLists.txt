add_subdirectory(array2d)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC array2d_lib)