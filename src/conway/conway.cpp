#include "conway.h"

#include <mpi.h>
#include <omp.h>

#include <array>
#include <filesystem>  // For std::filesystem
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "array2d.h"

using namespace conway;
using namespace array2d;

ConwaysArray2DWithHalo::ConwaysArray2DWithHalo(int n_rows, int n_cols)
    : Array2DWithHalo<int>(n_rows, n_cols){
          // Nothing required
      };

void ConwaysArray2DWithHalo::fill_randomly(float probability, int random_seed,
                                           bool fill_halo) {
    // Set the ranome seed appropriately
    int seed = [&random_seed]() -> int {
        if (random_seed == -1) {
            std::random_device rd;
            return rd();
        } else {
            return random_seed;
        }
    }();

    // Initialise random number generator with seed
    std::mt19937 gen(seed);

    // Initialise bernouli distribution with seed
    std::bernoulli_distribution dis(probability);

    int n_rows = this->get_rows();
    int n_cols = this->get_cols();

    if (!fill_halo) {
        // Fill in every cell value (excluding halo cells) with a 0 or 1
        for (int i = 0; i < n_rows; i++) {
            for (int j = 0; j < n_cols; j++) {
                (*this)(i, j) = dis(gen);
            }
        }
    } else {
        // Fill every cell value, including halo
        for (int i = -1; i < n_rows + 1; i++) {
            for (int j = -1; j < n_cols + 1; j++) {
                (*this)(i, j) = dis(gen);
            }
        }
    }
}

void ConwaysArray2DWithHalo::simple_convolve(array2d::Array2D<int> &neighbour_count) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
            int sum = (*this)(i, j - 1) + (*this)(i, j + 1) + (*this)(i - 1, j - 1) +
                      (*this)(i - 1, j) + (*this)(i - 1, j + 1) +
                      (*this)(i + 1, j - 1) + (*this)(i + 1, j) + (*this)(i + 1, j + 1);

            neighbour_count(i, j) = sum;
        }
    }
}

void ConwaysArray2DWithHalo::simple_convolve_inner(
    array2d::Array2D<int> &neighbour_count) {
#pragma omp parallel for collapse(2)
    for (int i = 1; i < this->n_rows - 1; i++) {
        for (int j = 1; j < this->n_cols - 1; j++) {
            int sum = (*this)(i, j - 1) + (*this)(i, j + 1) + (*this)(i - 1, j - 1) +
                      (*this)(i - 1, j) + (*this)(i - 1, j + 1) +
                      (*this)(i + 1, j - 1) + (*this)(i + 1, j) + (*this)(i + 1, j + 1);

            neighbour_count(i, j) = sum;
        }
    }
}

void ConwaysArray2DWithHalo::simple_convolve_outer(
    array2d::Array2D<int> &neighbour_count) {
#pragma omp parallel
    {
// Top row
#pragma omp for nowait
        for (int j = 0; j < n_cols; j++) {
            int i = 0;

            int sum = (*this)(i, j - 1) + (*this)(i, j + 1) + (*this)(i - 1, j - 1) +
                      (*this)(i - 1, j) + (*this)(i - 1, j + 1) +
                      (*this)(i + 1, j - 1) + (*this)(i + 1, j) + (*this)(i + 1, j + 1);

            neighbour_count(i, j) = sum;
        }

// Right column loop (excluding the first and last cell)
#pragma omp for nowait
        for (int i = 1; i < n_rows - 1; i++) {
            int j = this->n_cols - 1;

            int sum = (*this)(i, j - 1) + (*this)(i, j + 1) + (*this)(i - 1, j - 1) +
                      (*this)(i - 1, j) + (*this)(i - 1, j + 1) +
                      (*this)(i + 1, j - 1) + (*this)(i + 1, j) + (*this)(i + 1, j + 1);

            neighbour_count(i, j) = sum;
        }

// Left column loop (excluding the first and last cell)
#pragma omp for nowait
        for (int i = 1; i < n_rows - 1; i++) {
            int j = 0;

            int sum = (*this)(i, j - 1) + (*this)(i, j + 1) + (*this)(i - 1, j - 1) +
                      (*this)(i - 1, j) + (*this)(i - 1, j + 1) +
                      (*this)(i + 1, j - 1) + (*this)(i + 1, j) + (*this)(i + 1, j + 1);

            neighbour_count(i, j) = sum;
        }

// Bottom row
#pragma omp for
        for (int j = 0; j < n_cols; j++) {
            int i = n_rows - 1;

            int sum = (*this)(i, j - 1) + (*this)(i, j + 1) + (*this)(i - 1, j - 1) +
                      (*this)(i - 1, j) + (*this)(i - 1, j + 1) +
                      (*this)(i + 1, j - 1) + (*this)(i + 1, j) + (*this)(i + 1, j + 1);

            neighbour_count(i, j) = sum;
        }
    }
}

void ConwaysArray2DWithHalo::separable_convolution(
    array2d::Array2D<int> &neighbour_count) {
    array2d::Array2DWithHalo<int> horizontal_pass(this->n_rows, this->n_cols);

    // Horizontal pass: we need to loop over the halo row as well
#pragma omp parallel for collapse(2)
    for (int i = -1; i < this->n_rows + 1; i++) {
        for (int j = 0; j < this->n_cols; j++) {
            int sum = (*this)(i, j - 1) + (*this)(i, j) + (*this)(i, j + 1);
            horizontal_pass(i, j) = sum;
        }
    }

    // Vertical pass, save into neighbour_count
#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
            // also correct for the overcounting the middle cell
            int sum = horizontal_pass(i - 1, j) + horizontal_pass(i, j) +
                      horizontal_pass(i + 1, j) - (*this)(i, j);
            neighbour_count(i, j) = sum;
        }
    }
}

void ConwaysArray2DWithHalo::transition_ifs(array2d::Array2D<int> &neighbour_count) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
            int is_alive = (*this)(i, j);

            if (is_alive && neighbour_count(i, j) < 2) {
                (*this)(i, j) = 0;  // die by underpopulation
            } else if (is_alive && neighbour_count(i, j) > 3) {
                (*this)(i, j) = 0;  // die by overpopulation
            } else if (!is_alive && neighbour_count(i, j) == 3) {
                (*this)(i, j) = 1;  // reproduction
            }
        }
    }
}

void ConwaysArray2DWithHalo::transition_lookup(array2d::Array2D<int> &neighbour_count) {
    // lookup table is indexed by lookup_table[is_alive * 9 + neighbour_count(i, j)]
    std::array<int, 18> lookup_table = {
        0, 0, 0, 1, 0,
        0, 0, 0, 0,  // Dead cells spawn if they have 3 neighbours exactly

        0, 0, 1, 1, 0,
        0, 0, 0, 0  // Living cell stays alive with 2 or 3 live neighbors; otherwise,
                    // die
    };

#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
            int is_alive = (*this)(i, j);

            (*this)(i, j) = lookup_table[is_alive * 9 + neighbour_count(i, j)];
        }
    }
}

void ConwaysArray2DWithHalo::transition_bitwise(
    array2d::Array2D<int> &neighbour_count) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < this->n_rows; i++) {
        for (int j = 0; j < this->n_cols; j++) {
            int is_alive = (*this)(i, j);
            int count = neighbour_count(i, j);

            (*this)(i, j) =
                ((is_alive && (count == 2 || count == 3)) || (!is_alive && count == 3));
        }
    }
}

void ConwaysArray2DWithHalo::MPI_Isend_all(MPI_Comm &cartesian2d,
                                           std::array<MPI_Request, 8> &requests,
                                           std::array<int, 8> &neighbours,
                                           MPI_Datatype &MPI_Column_type) {
    /*
     * Here we go through each of the 8 possible directions we can send data:
     *   top-left, up, top-right, right, etc..
     * We send the data to the neighbour specified in the neighbours array.
     *
     * E.g., neighbours[0] should be the rank that is to the top left, so
     * that we can send the top left cell to the bottom right halo of the
     * top left rank.
     */
    MPI_Isend(&(*this)(0, 0), 1, MPI_INT, neighbours[0], 0, cartesian2d,
              &requests[0]);  // send top left cell

    MPI_Isend(&(*this)(0, 0), n_cols, MPI_INT, neighbours[1], 1, cartesian2d,
              &requests[1]);  // send top row

    MPI_Isend(&(*this)(0, n_cols - 1), 1, MPI_INT, neighbours[2], 2, cartesian2d,
              &requests[2]);  // send top right cell

    MPI_Isend(&(*this)(0, n_cols - 1), 1, MPI_Column_type, neighbours[3], 3,
              cartesian2d, &requests[3]);  // send right border

    MPI_Isend(&(*this)(n_rows - 1, n_cols - 1), 1, MPI_INT, neighbours[4], 4,
              cartesian2d, &requests[4]);  // send bottom right cell

    MPI_Isend(&(*this)(n_rows - 1, 0), n_cols, MPI_INT, neighbours[5], 5, cartesian2d,
              &requests[5]);  // send bottom row

    MPI_Isend(&(*this)(n_rows - 1, 0), 1, MPI_INT, neighbours[6], 6, cartesian2d,
              &requests[6]);  // send bottom left cell

    MPI_Isend(&(*this)(0, 0), 1, MPI_Column_type, neighbours[7], 7, cartesian2d,
              &requests[7]);  // send left border
}

void ConwaysArray2DWithHalo::MPI_Irecv_all(MPI_Comm &cartesian2d,
                                           std::array<MPI_Request, 8> &requests,
                                           std::array<int, 8> &neighbours,
                                           MPI_Datatype &MPI_Column_type) {
    /*
     * Here we go through each of the 8 possible directions we can recieve data
     * and receive it into the right section.
     *
     * E.g., neighbours[3] specifies the rank that is to the right of the
     * current rank. Therefore, we are expecting neighbours[3] to send
     * us a column of data, which we want to recieve into the right
     * halo column
     */
    MPI_Irecv(&(*this)(-1, -1), 1, MPI_INT, neighbours[0], 4, cartesian2d,
              &requests[0]);  // receive top left halo cell

    MPI_Irecv(&(*this)(-1, 0), n_cols, MPI_INT, neighbours[1], 5, cartesian2d,
              &requests[1]);  // receive top halo row

    MPI_Irecv(&(*this)(-1, n_cols), 1, MPI_INT, neighbours[2], 6, cartesian2d,
              &requests[2]);  // receive top right halo cell

    MPI_Irecv(&(*this)(0, n_cols), 1, MPI_Column_type, neighbours[3], 7, cartesian2d,
              &requests[3]);  // receive right halo column

    MPI_Irecv(&(*this)(n_rows, n_cols), 1, MPI_INT, neighbours[4], 0, cartesian2d,
              &requests[4]);  // receive bottom right halo cell

    MPI_Irecv(&(*this)(n_rows, 0), n_cols, MPI_INT, neighbours[5], 1, cartesian2d,
              &requests[5]);  // receive bottom halo row

    MPI_Irecv(&(*this)(n_rows, -1), 1, MPI_INT, neighbours[6], 2, cartesian2d,
              &requests[6]);  // receive bottom left halo cell

    MPI_Irecv(&(*this)(0, -1), 1, MPI_Column_type, neighbours[7], 3, cartesian2d,
              &requests[7]);  // receive left halo column
}

void ConwaysArray2DWithHalo::MPI_Wait_all(std::array<MPI_Request, 8> &send_requests,
                                          std::array<MPI_Request, 8> &recv_requests) {
    for (MPI_Request &req : send_requests) {
        MPI_Status status;
        MPI_Wait(&req, &status);
    }

    for (MPI_Request &req : recv_requests) {
        MPI_Status status;
        MPI_Wait(&req, &status);
    }
}

std::array<int, 8> ConwaysArray2DWithHalo::get_neighbour_ranks(
    MPI_Comm &cartesian2d, std::array<int, 2> &dims) {
    int rank, coords[2];
    MPI_Comm_rank(cartesian2d, &rank);
    MPI_Cart_coords(cartesian2d, rank, 2, coords);

    auto get_rank = [&cartesian2d, &dims](
                        const int coords[2],
                        const std::array<int, 2> &displacement) -> int {
        int neighbor_coords[2] = {(coords[0] + displacement[0]) % dims[0],
                                  (coords[1] + displacement[1]) % dims[1]};
        int neighbor_rank;
        MPI_Cart_rank(cartesian2d, neighbor_coords, &neighbor_rank);
        return neighbor_rank;
    };

    std::array<int, 8> neighbour_ranks = {
        get_rank(coords, {-1, -1}),  // top_left
        get_rank(coords, {-1, 0}),   // up
        get_rank(coords, {-1, 1}),   // top_right
        get_rank(coords, {0, 1}),    // right
        get_rank(coords, {1, 1}),    // bottom_right
        get_rank(coords, {1, 0}),    // down
        get_rank(coords, {1, -1}),   // bottom_left
        get_rank(coords, {0, -1})    // left
    };

    return neighbour_ranks;
}

std::array<int, 2> conway::get_decomposed_grid_size(int rank, int n_ranks,
                                                    int grid_size,
                                                    std::string decomposition_type) {
    if (decomposition_type == "column") {
        // Compute number of rows and columns for column-wise decompsition
        if (rank + 1 != n_ranks) {
            return {grid_size, grid_size / n_ranks};
        } else {
            // for the last rank, the number of columns may be different so the
            // below line accounts for the fact that grid_size may not be perfectly
            // divisible by n_ranks
            return {grid_size, grid_size - (n_ranks - 1) * (grid_size / n_ranks)};
        }
    } else if (decomposition_type == "row") {
        // Compute number of rows and columns for row-wise decompsition
        if (rank + 1 != n_ranks) {
            return {grid_size / n_ranks, grid_size};
        } else {
            return {grid_size - (n_ranks - 1) * (grid_size / n_ranks), grid_size};
        }
    } else {
        std::cerr
            << "Out of range: decomposition_type must be one of \"column\" or \"row\"."
            << std::endl;
        std::exit(1);
    }
}

void conway::read_from_text_file(conway::ConwaysArray2DWithHalo &arr,
                                 std::string filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open the file.");
    }

    std::string line;
    int row_count = 0;
    while (std::getline(file, line)) {
        for (int col = 0; col < line.size(); col++) {
            if (line[col] == '0' || line[col] == '1') {
                arr(row_count, col) = line[col] - '0';
            } else {
                // Handle error for unexpected characters
                throw std::runtime_error("Unexpected character in input file.");
            }
        }
        row_count++;
    }

    if (row_count != arr.get_rows()) {
        throw std::runtime_error(
            "The number of rows in the file does not match the expected size.");
    }
}

void conway::save_to_text_file(conway::ConwaysArray2DWithHalo &arr,
                               std::string filename) {
    std::ofstream file(filename);  // Open the file for writing

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
        return;
    }

    int n_rows = arr.get_rows();
    int n_cols = arr.get_cols();

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            file << std::to_string(arr(i, j));  // Write the element
        }
        file << "\n";  // End of row
    }

    file.close();  // Close the file
}

void conway::save_to_text_file(array2d::Array2D<int> &arr, std::string filename) {
    std::ofstream file(filename);  // Open the file for writing

    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
        return;
    }

    int n_rows = arr.get_rows();
    int n_cols = arr.get_cols();

    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            file << std::to_string(arr(i, j));  // Write the element
        }
        file << "\n";  // End of row
    }

    file.close();  // Close the file
}
