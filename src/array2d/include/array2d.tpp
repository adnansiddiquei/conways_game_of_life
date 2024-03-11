#include "array2d.h"
using namespace array2d;

// Constructor implementation
template <typename T>
Array2D<T>::Array2D(int n_rows, int n_cols) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    data = new T[n_rows * n_cols];
}

// Destructor implementation
template <typename T>
Array2D<T>::~Array2D() {
    delete[] data;
}

// Operator () overload implementation
template <typename T>
T& Array2D<T>::operator()(int i, int j) {
    return data[i * n_cols + j];
}

template <typename T>
T* Array2D<T>::get_pointer() {
    return data;
}

template <typename T>
int Array2D<T>::get_size() {
    return n_rows * n_cols;
}

template <typename T>
int Array2D<T>::get_rows() {
    return n_rows;
}

template <typename T>
int Array2D<T>::get_cols() {
    return n_cols;
}

// Transpose function implementation
template <typename T>
void array2d::transpose(Array2D<T>& arr_transposed, Array2D<T>& arr) {
    int n_rows = arr.get_rows(), n_cols = arr.get_cols();

    // transpose arr by writing contiguously
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; ++j) {
            arr_transposed(i, j) = arr(j, i);
        }
    }
};