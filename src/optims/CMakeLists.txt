add_executable(optims_convolutions convolutions.cpp)
ClangFormat(optims_convolutions .)

add_executable(optims_transitions transitions.cpp)
ClangFormat(optims_transitions .)

add_executable(optims_simpleio_convolution simpleio_convolution.cpp)
ClangFormat(optims_simpleio_convolution .)

add_executable(optims_lookup_transition lookup_transition.cpp)
ClangFormat(optims_lookup_transition .)

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

target_link_libraries(optims_simpleio_convolution 
    PUBLIC 
    array2d_lib 
    conway_lib 
    timer_lib
)

target_link_libraries(optims_lookup_transition 
    PUBLIC 
    array2d_lib 
    conway_lib 
    timer_lib
)