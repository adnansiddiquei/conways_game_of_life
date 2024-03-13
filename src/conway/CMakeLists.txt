add_library(conway_lib
    STATIC
    conway.cpp
)

target_include_directories(conway_lib
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(conway_lib PUBLIC array2d_lib)