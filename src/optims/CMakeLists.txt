add_executable(convolutions convolutions.cpp)
ClangFormat(convolutions .)

add_executable(transitions transitions.cpp)
ClangFormat(transitions .)

target_link_libraries(convolutions 
    PUBLIC 
    array2d_lib 
    conway_lib 
    timer_lib
)

target_link_libraries(transitions 
    PUBLIC 
    array2d_lib 
    conway_lib 
    timer_lib
)