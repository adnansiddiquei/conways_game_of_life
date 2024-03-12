#ifndef AS3438_ARRAY2D_H
#define AS3438_ARRAY2D_H

namespace array2d {
/**
 * @class Array2D
 * @brief Implements 2D array template class.
 *
 * This class implements a 2D array by implementing it as a 1D array under the
 * hood with 2D indexing.
 *
 * @tparam T The type of the array elements.
 */
template <typename T>
class Array2D {
   protected:
    int n_rows;  ///< Number of rows.
    int n_cols;  ///< Number of columns.
    T *data;     ///< Pointer to the array data.

   public:
    /**
     * Constructor that initialises the 2D array to the specified dimensions and
     * allocates memory for the array elements.
     *
     * @param n_rows Number of rows in array.
     * @param n_cols Number of columns in array.
     */
    Array2D(int n_rows, int n_cols);

    /**
     * Class destructor, frees the array from memory when it is no longer needed.
     */
    ~Array2D();

    /**
     * Overloads the function call operator to provide access to the array
     * elements using a 2D indexing notation.
     *
     * @param i Row index.
     * @param j Column index.
     *
     * @return Reference to element as the specified location `array(i, j)`.
     */
    T &operator()(int i, int j);

    /**
     * Returns pointer to the underlying array data.
     *
     * @return Pointer to the array data.
     */
    T *get_pointer();

    /**
     * Calculates the total number of elements in the array.
     *
     * @return Total number of elements in the array.
     */
    int get_size();

    /**
     * Returns the number of rows.

     * @return Number of rows.
     */
    int get_rows();

    /**
     * Returns the number os columns.
     *
     * @return Number of columns.
     */
    int get_cols();
};

// TODO: comment this code
template <typename T>
class Array2DWithHalo : public Array2D<T> {
   private:
    int halo_size;  ///< Number of cells in halo.

   public:
    Array2DWithHalo(int n_rows, int n_cols, int halo_size);

    T &operator()(int i, int j);
};

/**
 * Transposes `arr` and save it into `arr_transposed`.
 *
 * It is expected that both `arr` and `arr_transposed` are of the dimensions, for speed,
 * no checks are done to ensure that they of the same dimensions.
 *
 * @tparam T The type of the elements in `arr` and `arr_transposed`.
 * @param arr_transposed The array to save the transpose into.
 * @param arr The array to transpose.
 */
template <typename T>
void transpose(Array2D<T> &arr_transposed, Array2D<T> &arr);

}  // namespace array2d

#include "array2d.tpp"

#endif  // AS3438_ARRAY2D_H
