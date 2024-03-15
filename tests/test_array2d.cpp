/**
 * @file test_array2d.cpp
 * @brief This file contains all the test cases for the Array2D class.
 */

#include <gtest/gtest.h>

#include "../src/array2d/include/array2d.h"

/**
 * Test array2d::Array2D
 */
TEST(array2D, Array2D) {
    int n_rows = 10, n_cols = 12, intial_value = 22;

    // Basic operations
    array2d::Array2D<int> arr(n_rows, n_cols, intial_value);

    EXPECT_EQ(arr.get_rows(), n_rows);
    EXPECT_EQ(arr.get_cols(), n_cols);
    EXPECT_EQ(arr.get_size(), n_rows * n_cols);

    // Make sure the initial value works
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            EXPECT_EQ(arr(i, j), intial_value);
        }
    }

    arr(7, 2) = 43;
    EXPECT_EQ(arr(7, 2), 43);
}

/**
 * Test array2d::Array2DWithHalo
 */
TEST(array2D, Array2DWithHalo) {
    int n_rows = 8, n_cols = 16;

    // Basic operations
    array2d::Array2DWithHalo<int> arr(n_rows, n_cols);

    EXPECT_EQ(arr.get_rows(), n_rows);
    EXPECT_EQ(arr.get_cols(), n_cols);
    EXPECT_EQ(arr.get_size(), n_rows * n_cols);

    arr(7, 2) = 22;
    EXPECT_EQ(arr(7, 2), 22);

    // Make sure there is a halo
    for (int i = -1; i < n_rows + 1; i++) {
        for (int j = -1; j < n_cols + 1; j++) {
            arr(i, j) = 67;
            EXPECT_EQ(arr(i, j), 67);
        }
    }
}
