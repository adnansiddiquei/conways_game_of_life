/**
 * @file test_array2d.cpp
 * @brief This file contains all the test cases for the Array2D class.
 */

#include <gtest/gtest.h>

#include "../src/array2d/include/array2d.h"

/**
 * Tests basic array operations such as creating, editing and using class methods.
 */
TEST(array2d, array_operations) {
    array2d::Array2D<int> arr(10, 10);
    arr(7, 2) = 43;

    EXPECT_EQ(arr.get_cols(), 10);
    EXPECT_EQ(arr.get_rows(), 10);
    EXPECT_EQ(arr.get_size(), 100);
    EXPECT_EQ(arr(7, 2), 43);
}

/**
 * Tests the transpose operation array2d::transpose.
 */
TEST(array2d, transpose_operation) {
    array2d::Array2D<int> arr(4, 4);
    array2d::Array2D<int> arr_transposed(4, 4);

    arr(0, 1) = 2;
    arr(1, 0) = 5;
    arr(2, 2) = 8;
    arr(3, 1) = 13;
    arr(3, 2) = 15;

    array2d::transpose(arr_transposed, arr);

    // Check if the transpose operation was successful
    EXPECT_EQ(arr_transposed(1, 0), 2);
    EXPECT_EQ(arr_transposed(0, 1), 5);
    EXPECT_EQ(arr_transposed(2, 2), 8);
    EXPECT_EQ(arr_transposed(1, 3), 13);
    EXPECT_EQ(arr_transposed(2, 3), 15);
}