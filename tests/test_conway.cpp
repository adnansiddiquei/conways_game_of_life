#include <gtest/gtest.h>

#include <filesystem>  // For std::filesystem

#include "array2d.h"
#include "conway.h"

/**
 * @brief  Function that ensures an arrays values are between a min (inclusive) and max
 (exclusive).

 * @param  array: The Array2D to test.
 * @param  min_val: Min values (inclusive).
 * @param  max_val: Max value (excludive).
 */
template <typename T>
void check_array_elements_in_range(array2d::Array2D<T>& array, T min_val, T max_val) {
    for (int i = 0; i < array.get_rows(); i++) {
        for (int j = 0; j < array.get_cols(); j++) {
            EXPECT_GE(array(i, j), min_val);
            EXPECT_LT(array(i, j), max_val);
        }
    }
}

/**
 * Test conway::ConwaysArray2DWithHalo
 */
TEST(conway, ConwaysArray2DWithHalo) {
    int n_rows = 10, n_cols = 12;

    // Basic operations
    conway::ConwaysArray2DWithHalo arr(n_rows, n_cols);

    EXPECT_EQ(arr.get_rows(), n_rows);
    EXPECT_EQ(arr.get_cols(), n_cols);
    EXPECT_EQ(arr.get_size(), n_rows * n_cols);

    // Make sure assignment works
    arr(7, 2) = 43;
    EXPECT_EQ(arr(7, 2), 43);
}

/**
 * Test conway::ConwaysArray2DWithHalo.fill_randomly
 */
TEST(conway, ConwaysArray2DWithHalo__fill_randomly) {
    int n_rows = 10, n_cols = 12;

    conway::ConwaysArray2DWithHalo arr1(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo arr2(n_rows, n_cols);

    arr1.fill_randomly(0.6, 42, false);
    arr2.fill_randomly(0.6, 42, false);

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            EXPECT_EQ(arr1(i, j), arr2(i, j));
        }
    }

    conway::ConwaysArray2DWithHalo arr3(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo arr4(n_rows, n_cols);

    arr1.fill_randomly(0.6, 42, true);
    arr2.fill_randomly(0.6, 42, true);

    for (int i = -1; i < n_rows + 1; i++) {
        for (int j = -1; j < n_cols + 1; j++) {
            EXPECT_EQ(arr1(i, j), arr2(i, j));
        }
    }
}

/**
 * Test conway::ConwaysArray2DWithHalo.simple_convolve and
 * conway::ConwaysArray2DWithHalo.separable_convolution and
 * conway::ConwaysArray2DWithHalo.simple_convolve_inner() followed by
 * conway::ConwaysArray2DWithHalo.simple_convolve_outer().
 */
TEST(conway, ConwaysArray2DWithHalo__convolve) {
    int n_rows = 10, n_cols = 12;

    conway::ConwaysArray2DWithHalo arr1(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo arr2(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo arr3(n_rows, n_cols);

    // fill both arrays identically
    arr1.fill_randomly(0.6, 42, true);
    arr2.fill_randomly(0.6, 42, true);
    arr3.fill_randomly(0.6, 42, true);

    // Make sure that all cells are between [0, 1)
    check_array_elements_in_range(arr1, 0, 2);
    check_array_elements_in_range(arr2, 0, 2);
    check_array_elements_in_range(arr3, 0, 2);

    // Count the neighbours using the two different convolution methods
    array2d::Array2D<int> neighbour_count1(n_rows, n_cols);
    array2d::Array2D<int> neighbour_count2(n_rows, n_cols);
    array2d::Array2D<int> neighbour_count3(n_rows, n_cols);

    arr1.simple_convolve(neighbour_count1);        // simple convolve
    arr2.separable_convolution(neighbour_count2);  // separable convolution
    arr3.simple_convolve_inner(neighbour_count3);  // inner then outer ring
    arr3.simple_convolve_outer(neighbour_count3);

    // Make sure that they both give the same result, as they should
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            EXPECT_EQ(neighbour_count1(i, j), neighbour_count2(i, j));
            EXPECT_EQ(neighbour_count2(i, j), neighbour_count3(i, j));
        }
    }
}

/**
 * Test conway::ConwaysArray2DWithHalo.transition_ifs and
 * conway::ConwaysArray2DWithHalo.transition_lookup and
 * conway::ConwaysArray2DWithHalo.transition_bitwise.
 */
TEST(conway, ConwaysArray2DWithHalo__transition) {
    int n_rows = 10, n_cols = 12;

    conway::ConwaysArray2DWithHalo arr1(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo arr2(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo arr3(n_rows, n_cols);

    // fill the arrays identically
    arr1.fill_randomly(0.6, 42, true);
    arr2.fill_randomly(0.6, 42, true);
    arr3.fill_randomly(0.6, 42, true);

    // Count the neighbours
    array2d::Array2D<int> neighbour_count1(n_rows, n_cols);
    array2d::Array2D<int> neighbour_count2(n_rows, n_cols);
    array2d::Array2D<int> neighbour_count3(n_rows, n_cols);

    arr1.simple_convolve(neighbour_count1);
    arr2.simple_convolve(neighbour_count2);
    arr3.simple_convolve(neighbour_count3);

    // Make sure that all methods gave the same neighbour count
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            EXPECT_EQ(neighbour_count1(i, j), neighbour_count2(i, j));
            EXPECT_EQ(neighbour_count2(i, j), neighbour_count3(i, j));
        }
    }

    // Make sure that all neighbour counts are in the appropriate range
    check_array_elements_in_range(neighbour_count1, 0, 9);
    check_array_elements_in_range(neighbour_count2, 0, 9);
    check_array_elements_in_range(neighbour_count3, 0, 9);

    // Now progress the arrays onto the next generation using the 3 methods
    arr1.transition_ifs(neighbour_count1);
    arr2.transition_lookup(neighbour_count2);
    arr3.transition_bitwise(neighbour_count3);

    // Make sure they all progressed exactly the same
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            EXPECT_EQ(arr1(i, j), arr2(i, j));
            EXPECT_EQ(arr2(i, j), arr3(i, j));
        }
    }
}

/**
 * Test conway::read_from_text_file
 */
TEST(conway, read_from_text_file) {
    int n_rows = 81, n_cols = 81;

    conway::ConwaysArray2DWithHalo arr(n_rows, n_cols);
    conway::read_from_text_file(
        arr, std::filesystem::absolute("../../tests/test_grid_glider.txt"));

    ASSERT_EQ(arr(0, 1), 1);
    ASSERT_EQ(arr(1, 2), 1);
    ASSERT_EQ(arr(2, 0), 1);
    ASSERT_EQ(arr(2, 1), 1);
    ASSERT_EQ(arr(2, 2), 1);

    arr(0, 1) = 0;
    arr(1, 2) = 0;
    arr(2, 0) = 0;
    arr(2, 1) = 0;
    arr(2, 2) = 0;

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            ASSERT_EQ(arr(i, j), 0);
        }
    }
}

/**
 * Test conway::save_to_text_file
 */
TEST(conway, save_to_text_file) {
    int n_rows = 81, n_cols = 81;

    conway::ConwaysArray2DWithHalo arr(n_rows, n_cols);
    conway::read_from_text_file(
        arr, std::filesystem::absolute("../../tests/test_grid_glider.txt"));

    ASSERT_EQ(arr(0, 1), 1);
    ASSERT_EQ(arr(1, 2), 1);
    ASSERT_EQ(arr(2, 0), 1);
    ASSERT_EQ(arr(2, 1), 1);
    ASSERT_EQ(arr(2, 2), 1);

    conway::save_to_text_file(
        arr, std::filesystem::absolute("../../bin/test_grid_glider_test_output.txt"));

    conway::ConwaysArray2DWithHalo arr2(n_rows, n_cols);
    conway::read_from_text_file(
        arr2, std::filesystem::absolute("../../bin/test_grid_glider_test_output.txt"));

    ASSERT_EQ(arr2(0, 1), 1);
    ASSERT_EQ(arr2(1, 2), 1);
    ASSERT_EQ(arr2(2, 0), 1);
    ASSERT_EQ(arr2(2, 1), 1);
    ASSERT_EQ(arr2(2, 2), 1);

    std::filesystem::remove(
        std::filesystem::absolute("../../bin/test_grid_glider_test_output.txt"));
}