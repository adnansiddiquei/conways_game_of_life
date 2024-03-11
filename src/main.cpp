#include "array2d.h"
#include <iostream>


int main(int argc, char *argv[]) {
    array2d::Array2D<int> arr(10, 10);
    array2d::Array2D<int> arr_transposed(10, 10);
    arr(0, 1) = 511;

    std::cout << arr(0, 1) << std::endl;

    array2d::transpose(arr_transposed, arr);

    std::cout << arr_transposed(1, 0) << std::endl;

    return 0;
}