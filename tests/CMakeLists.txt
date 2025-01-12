include(FetchContent)

# Get GoogleTests
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.13.0
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest\_force\_shared\_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add the test_array2d tests and also format the file with ClangFormat
add_executable(test_array2d test_array2d.cpp)
ClangFormat(test_array2d .)

target_link_libraries(test_array2d
    PRIVATE array2d_lib
    GTest::gtest_main
)

# Add the test_conway tests and also format the file with ClangFormat
add_executable(test_conway test_conway.cpp)
ClangFormat(test_conway .)

target_link_libraries(test_conway
    PRIVATE conway_lib
    GTest::gtest_main
)

# Add tests for test_mpi_comms
add_executable(test_mpi_comms test_mpi_comms.cpp)
ClangFormat(test_mpi_comms .)

target_link_libraries(test_mpi_comms
    PRIVATE conway_lib
    GTest::gtest_main
)

include(GoogleTest)

gtest_discover_tests(test_array2d)
gtest_discover_tests(test_conway)

