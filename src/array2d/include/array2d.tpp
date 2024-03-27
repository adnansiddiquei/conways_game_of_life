#include "array2d.h"
using namespace array2d;

/**
 * Implementation for Array2D class.
 */

// Constructor implementation
template <typename T>
Array2D<T>::Array2D(int n_rows, int n_cols) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    data = new T[n_rows * n_cols];
}

template <typename T>
Array2D<T>::Array2D(int n_rows, int n_cols, T initial_value) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
    data = new T[n_rows * n_cols];

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            data[i * n_cols + j] = initial_value;
        }
    }
}

// Destructor implementation
template <typename T>
Array2D<T>::~Array2D() {
    delete[] data;
}

// Operator () overload implementation
template <typename T>
T &Array2D<T>::operator()(int i, int j) {
    return data[i * n_cols + j];
}

template <typename T>
T *Array2D<T>::get_pointer() {
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

/**
 * Implementation for Array2DWithHalo class.
 */

template <typename T>
Array2DWithHalo<T>::Array2DWithHalo(int n_rows, int n_cols)
    : Array2D<T>(n_rows + 2, n_cols + 2) {
    this->n_rows = n_rows;
    this->n_cols = n_cols;
};

template <typename T>
T &Array2DWithHalo<T>::operator()(int i, int j) {
    // Indexing (0, 0) starts from the top left cell that is not a halo
    return this->data[(i + 1) * (this->n_cols + 2) + (j + 1)];
};
