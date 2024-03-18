add_subdirectory(array2d)
add_subdirectory(conway)
add_subdirectory(timer)
add_subdirectory(optims)

add_executable(main main.cpp)
ClangFormat(main .)

target_link_libraries(main PUBLIC array2d_lib)
target_link_libraries(main PUBLIC conway_lib)
target_link_libraries(main PUBLIC timer_lib)
target_link_libraries(main PUBLIC MPI::MPI_CXX)