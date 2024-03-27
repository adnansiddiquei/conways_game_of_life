#ifndef AS3438_ARRAY2D_H
#define AS3438_ARRAY2D_H

namespace array2d {
/**
 * @class Array2D
 * @brief Implements 2D array template class.
 *
 * This class implements a 2D array by implementing it as a 1D array in
 * row-major order.
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
     * @brief Constructor that initialises the 2D array to the specified dimensions and
     * allocates memory for the array elements.
     *
     * @param n_rows Number of rows in array.
     * @param n_cols Number of columns in array.
     */
    Array2D(int n_rows, int n_cols);

    /**
     * @brief Constructor that initialises the 2D array to the specified dimensions and
     * allocates memory for the array elements, and sets the value for each cell.
     *
     * @param n_rows Number of rows in array.
     * @param n_cols Number of columns in array.
     */
    Array2D(int n_rows, int n_cols, T initial_value);

    /**
     * @brief Class destructor, frees the array from memory when it is no longer needed.
     */
    virtual ~Array2D();

    /**
     * @brief Overloads the function call operator to provide access to the array
     * elements using a 2D indexing notation.
     *
     * @param i Row index.
     * @param j Column index.
     *
     * @return Reference to element as the specified location `array(i, j)`.
     */
    T &operator()(int i, int j);

    /**
     * @brief Returns pointer to the underlying array data.
     *
     * @return Pointer to the array data.
     */
    T *get_pointer();

    /**
     * @brief Calculates the total number of elements in the array.
     *
     * @return Total number of elements in the array.
     */
    int get_size();

    /**
     * @brief Returns the number of rows.

     * @return Number of rows.
     */
    int get_rows();

    /**
     * @brief Returns the number os columns.
     *
     * @return Number of columns.
     */
    int get_cols();
};

/**
 * @class Array2DWithHalo
 * @brief Implements a 2DArray with a Halo.
 *
 * This class inherits from `Array2D` and implements a grid that has a 1 cell
 * halo around it. The only difference between `Array2D` and `Array2DWithHalo`
 * is that this class re-implements the constructor and `()` operator such that
 * when the class is initialised, a 1 cell halo is created around the array,
 * and indexing with (i,j) will return to you the cell that with that
 * co-ordinate inside the domain rather than the halo.
 *
 * I.e.,
 *      (-1, -1) returns the top left halo;
 *      (0, 0) returns the top left cell, not in the halo
 *      (n_rows, n_cols) returns the bottom right halo
 *
 * @tparam T The type of the array elements.
 */
template <typename T>
class Array2DWithHalo : public Array2D<T> {
   public:
    /**
     * @brief Constructor that initialises the 2D array to the specified dimensions and
     * allocates memory for the array elements, with a 1-cell halo.
     *
     * @param  n_rows: Number of rows (not including the 2 halo rows).
     * @param  n_cols: Number of columns (not including the 2 halo columns).
     */
    Array2DWithHalo(int n_rows, int n_cols);

    /**
     * @brief Overloads the function call operator to provide access to the array
     * elements using a 2D indexing notation.
     *
     * The top left halo is accessed with (-1,-1). The top left non-halo
     * cell is accessed by (0,0).
     *
     * @param i Row index (starting at the first non-halo row).
     * @param j Column index (starting at the first non-halo column).
     *
     * @return Reference to element as the specified location `array(i, j)`.
     */
    T &operator()(int i, int j);
};

}  // namespace array2d

#include "array2d.tpp"

#endif  // AS3438_ARRAY2D_H
