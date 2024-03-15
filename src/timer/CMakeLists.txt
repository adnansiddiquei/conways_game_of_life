add_library(timer_lib
    STATIC 
    timer.cpp
)

target_include_directories(timer_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
