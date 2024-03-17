add_executable(optims_convolutions convolutions.cpp)
ClangFormat(optims_convolutions .)

add_executable(optims_transitions transitions.cpp)
ClangFormat(optims_transitions .)

target_link_libraries(optims_convolutions 
    PUBLIC 
    array2d_lib 
    conway_lib 
    timer_lib
)

target_link_libraries(optims_transitions 
    PUBLIC 
    array2d_lib 
    conway_lib 
    timer_lib
)