#include <mpi.h>

#include <array>
#include <fstream>
#include <string>

#include "array2d.h"

#ifndef AS3438_CONWAY_H
#define AS3438_CONWAY_H

// TODO: comment
namespace conway {
class ConwaysArray2DWithHalo : public array2d::Array2DWithHalo<int> {
   public:
    /**
     * @brief  Constructor for ConwaysArray2DWithHalo.
     * @note
     * @param  n_rows: Number of rows (not including the 2 halo rows).
     * @param  n_cols: Number of columns (not including the 2 halo columns).
     * @retval
     */
    ConwaysArray2DWithHalo(int n_rows, int n_cols);

    /**
     * @brief  Fill the array randomly with 0s and 1s with a given probability.
     *
     * @param  probability: The probability that a cell will be 1.
     * @param  random_seed: Random seed, for reproducibility.
     * @param  fill_halo: Whether to fill the halo with 1s and 0s as well of not.
     */
    void fill_randomly(float probability, int random_seed = -1, bool fill_halo = false);

    /**
     * @brief  Count the neighbours of every cell using a simple convolution
     * with a 3x3 kernel.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. The neighbour counts will
     * be stored in this array.
     */
    void simple_convolve(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  Count the neighbours of every cell using the separable convolution
     * method.
     *
     * This involves doing a horizontal pass with the kernel [1,1,1], and then
     * doing a vertical pass with the kernel [1,1,1]^T, followed by minusing off
     * the current value in each cell.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. The neighbour counts will
     * be stored in this array.
     */
    void separable_convolution(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  A simple convolution like `simple_convolve`, but only the neighbours
     * of cells that ARE NOT adjacent to halo cells are counted.
     *
     * This is part 1 of a full simple convolution. This function can count
     * neighbours without the need for the halos to be populated as it does
     * not count the neighbours for cells that are adjacent to halo cells.
     * To finish counting the neighbours, you would need to call the
     * `simple_convolve_inner` function and pass in the same `&neighbour_count`
     * array.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. The neighbour counts will
     * be stored in this array.
     */
    void simple_convolve_inner(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  A simple convolution like `simple_convolve`, but only the neighbours
     * of cells that ARE adjacent to halo cells are counted.
     *
     * This is part 2 of a full simple convolve. To count the inner neighbours,
     * use `simple_convolve_inner`.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. The neighbour counts will
     * be stored in this array.
     */
    void simple_convolve_outer(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  Transition the grid to the next generation using `if` statements
     * to assess the fate of each cell.
     *
     * After this function runs, the cells in the data array will all be evolved
     * into the next generation based on their neighbour counts.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. This must be populated with
     * the neighbour counts of the corresponding cells, ranging from [0, 9].
     */
    void transition_ifs(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  Transition the grid to the next generation using the lookup array
     * method to assess the fate of each cell.
     *
     * After this function runs, the cells in the data array will all be evolved
     * into the next generation based on their neighbour counts.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. This must be populated with
     * the neighbour counts of the corresponding cells, ranging from [0, 9].
     */
    void transition_lookup(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  Transition the grid to the next generation using the bitwise
     * method to assess the fate of each cell.
     *
     * After this function runs, the cells in the data array will all be evolved
     * into the next generation based on their neighbour counts.
     *
     * @param  &neighbour_count: An Array2D with the same dimensions as the
     * the current ConwaysArray2DWithHalo instance. This must be populated with
     * the neighbour counts of the corresponding cells, ranging from [0, 9].
     */
    void transition_bitwise(array2d::Array2D<int> &neighbour_count);

    /**
     * @brief  Send all of the border cells to the halos cells of the adjacent
     * MPI ranks in a 2D cartesian topology.
     *
     * @param  &cartesian2d: The MPI_Comm variable associated wth the cartesian2d.
     * @param  &requests: An array of MPI_Request objects. These are used to track the
     * status of the non-blocking send operations. There are 8 requests corresponding to
     * the 8 possible directions of communication in a 2D grid (including diagonals).
     * @param  &neighbours: An array of integers holding the rank IDs of the neighbour
     * ranks in the cartesian2d. The order goes: top left, top, top right, right, bottom
     * right, bottom, bottom left, left.
     * @param  &MPI_Column_type: The MPI_Datatype that describes the layout of the
     * column data that is being sent to and from the horizonal borders.
     */
    void MPI_Isend_all(MPI_Comm &cartesian2d, std::array<MPI_Request, 8> &requests,
                       std::array<int, 8> &neighbours, MPI_Datatype &MPI_Column_type);

    void MPI_Irecv_all(MPI_Comm &cartesian2d, std::array<MPI_Request, 8> &requests,
                       std::array<int, 8> &neighbours, MPI_Datatype &MPI_Column_type);

    void MPI_Wait_all(std::array<MPI_Request, 8> &send_reqs,
                      std::array<MPI_Request, 8> &recv_reqs);

    std::array<int, 8> get_neighbour_ranks(MPI_Comm &cartesian2d,
                                           std::array<int, 2> &dims);
};

std::array<int, 2> get_decomposed_grid_size(int rank, int n_ranks, int grid_size,
                                            std::string decomposition_type);

void read_from_text_file(conway::ConwaysArray2DWithHalo &arr, std::string filename);

void save_to_text_file(array2d::Array2D<int> &arr, std::string filename);

void save_to_text_file(conway::ConwaysArray2DWithHalo &arr, std::string filename);

}  // namespace conway

#endif  // AS3438_CONWAY_H
