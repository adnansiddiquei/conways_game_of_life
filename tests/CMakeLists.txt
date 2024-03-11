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

include(GoogleTest)
gtest_discover_tests(test_array2d)

