add_subdirectory(utils)

add_executable(main main.cpp)

target_link_libraries(main PUBLIC utils_lib)