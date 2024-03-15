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
