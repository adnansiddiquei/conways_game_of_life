#include <mpi.h>

#include <array>
#include <cstdlib>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

#include "array2d.h"
#include "conway.h"

/**
 * Parses the command line arguments passed into this script, and returns the value (as
 * a std::string) of the requested option.
 *
 * An empty string "" is returned if the option is not required and was not provided.
 *
 * @param argc The count of arguments passed into the script.
 * @param argv The values passed into the script.
 * @param option The option to retrieve the value for.
 * @param required Whether the specified option is required. If it is required, the
 * program will exit with an appropriate error message.
 *
 * @return The value of the requested option.
 */
std::string parse_arg(int &argc, char *argv[], std::string option, bool required) {
    for (int i = 0; i < argc; i++) {
        std::string current_option = argv[i];

        if (current_option == option) {
            if (i + 1 < argc) {
                return argv[i + 1];
            } else {
                std::cerr << option << " option requires one argument." << std::endl;
                std::exit(1);
            }
        }
    }

    // if code reaches here then we are looking for an argument that was not passed into
    // the terminal
    if (required) {
        // exit with an error if the option is required
        std::string errorMessage = option + " option must be provided.";
        std::cerr << errorMessage << std::endl;
        std::exit(1);
    } else {
        return "";
    }
}

/**
 * Parses the command line arguments passed into this script, and returns the value (as
 * an int) of the requested option.
 *
 * `-1` is returned if the option is not required and was not provided.
 *
 * @param argc The count of arguments passed into the script.
 * @param argv The values passed into the script.
 * @param option The option to retrieve the value for.
 * @param required Whether the specified option is required. If it is required, the
 * program will exit with an appropriate error message.
 *
 * @return The value (as an int) of the requested option.
 *
 */
int parse_arg_int(int &argc, char *argv[], std::string option, bool required) {
    std::string arg = parse_arg(argc, argv, option, required);

    // If the option is not required and has not been provided, return a -1.
    if (!required & arg == "") {
        return -1;
    }

    // try a type conversion to int, but raise the  appropriate error if this fails
    try {
        return std::stoi(arg);
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << option << " argument must an integer."
                  << std::endl;
        std::exit(1);
    } catch (const std::out_of_range &e) {
        // Handle case where the string represents a number outside the range of int
        std::cerr << "Out of range: " << option << " argument is too large."
                  << std::endl;
        std::exit(1);
    }
}

/**
 * Parses the command line arguments passed into this script, and returns the value (as
 * a float) of the requested option.
 *
 * `-1.` is returned if the option is not required and was not provided.
 *
 * @param argc The count of arguments passed into the script.
 * @param argv The values passed into the script.
 * @param option The option to retrieve the value for.
 * @param required Whether the specified option is required. If it is required, the
 * program will exit with an appropriate error message.
 *
 * @return The value (as a float) of the requested option.
 *
 */
float parse_arg_float(int &argc, char *argv[], std::string option, bool required) {
    std::string arg = parse_arg(argc, argv, option, required);

    // If the option is not required and has not been provided, return a -1.
    if (!required & arg == "") {
        return -1.;
    }

    // try a type conversion to float, but raise the appropriate error if this fails
    try {
        return std::stof(arg);
    } catch (const std::invalid_argument &e) {
        std::cerr << "Invalid argument: " << option << " argument must a float."
                  << std::endl;
        std::exit(1);
    } catch (const std::out_of_range &e) {
        // Handle case where the string represents a number outside the range of int
        std::cerr << "Out of range: " << option << " argument is too large."
                  << std::endl;
        std::exit(1);
    }
}

int get_neighbor_rank(MPI_Comm &cartesian2d, const std::array<int, 2> &dims,
                      int disp_row, int disp_col) {
    int rank, coords[2], neighbor_rank;
    MPI_Comm_rank(cartesian2d, &rank);
    MPI_Cart_coords(cartesian2d, rank, 2, coords);

    // Adjust coordinates to get to the co-ordinates of the desired rank
    coords[0] = (coords[0] + disp_row + dims[0]) % dims[0];
    coords[1] = (coords[1] + disp_col + dims[1]) % dims[1];

    // get the rank number of the rank at the desired co-ords, and return
    MPI_Cart_rank(cartesian2d, coords, &neighbor_rank);
    return neighbor_rank;
}

bool DEBUG_MODE = true;

void logger(std::string text) {
    if (DEBUG_MODE) {
        std::cout << text << std::endl;
    }
}

int main(int argc, char *argv[]) {
    // parse the arguments passed into the script
    int grid_size = parse_arg_int(argc, argv, "--grid-size", true);
    int random_seed = parse_arg_int(argc, argv, "--random-seed", false);
    float probability = parse_arg_float(argc, argv, "--probability", true);
    std::string decomposition_type = parse_arg(argc, argv, "--mpi-decomp", true);

    // Ensure that a valid decomposition type was specified
    if (decomposition_type != "column" & decomposition_type != "row" &
        decomposition_type != "grid") {
        std::cerr << "Out of range: --mpi-decomp must be one of \"column\", \"row\" or "
                     "\"grid\"."
                  << std::endl;
        std::exit(1);
    }

    // Now we create the MPI ranks and set up the cartesian communicator
    MPI_Init(&argc, &argv);

    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // settings for the cartesian2d communicator
    std::array<int, 2> dims = [&decomposition_type, &n_ranks]() -> std::array<int, 2> {
        if (decomposition_type == "column") {
            return {1, n_ranks};  // column-wise decomposition
        } else if (decomposition_type == "row") {
            return {n_ranks, 1};  // row-wise decomposition
        }

        return {0, 0};
    }();

    std::array<int, 2> periods = {1, 1};  // periodic boundary conditions in both axes
    int reorder = 1;

    // create the cartesian2d communicator
    MPI_Comm cartesian2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), reorder,
                    &cartesian2d);

    // Get the grid size of the decomposed grid
    /**
     * TODO: In the case that grid_size / n_ranks == 0.
     * This code needs to be edited to handle when the grid might be very small.
     * Maybe in that case just don't even bother doing domain decomposition.
     */
    std::array<int, 2> decomposed_grid_size =
        [&rank, &n_ranks, &grid_size, &decomposition_type]() -> std::array<int, 2> {
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
        }

        return {0, 0};
    }();

    int n_rows = decomposed_grid_size[0];
    int n_cols = decomposed_grid_size[1];

    // Create the 2D array that will represent the decomposed domain
    conway::ConwaysArray2DWithHalo grid(n_rows, n_cols);

    // Fill the grid (excluding the halo) with 1s and 0s according to the probability
    // and random_seed the user inputted on the command line.
    grid.fill_randomly(probability, random_seed);

    /**
     * This section below is simply calculating which rank is in each direction
     * of the current rank.
     *
     * `left` will be an int signifying the rank that is to the left of the current
     * rank. `top_left` will signify the rank that it one to the left and one up.
     *
     */

    int left, right, up, down, top_left, top_right, bottom_left, bottom_right;

    left = get_neighbor_rank(cartesian2d, dims, 0, -1);
    right = get_neighbor_rank(cartesian2d, dims, 0, 1);
    up = get_neighbor_rank(cartesian2d, dims, -1, 0);
    down = get_neighbor_rank(cartesian2d, dims, 1, 0);
    top_left = get_neighbor_rank(cartesian2d, dims, -1, -1);
    top_right = get_neighbor_rank(cartesian2d, dims, -1, 1);
    bottom_left = get_neighbor_rank(cartesian2d, dims, 1, -1);
    bottom_right = get_neighbor_rank(cartesian2d, dims, 1, 1);

    logger("Rank " + std::to_string(rank) + ": " + std::to_string(n_rows) + " x " +
           std::to_string(n_cols) + ". " + std::to_string(left) +
           std::to_string(right) + std::to_string(up) + std::to_string(down) +
           std::to_string(top_left) + std::to_string(top_right) +
           std::to_string(bottom_left) + std::to_string(bottom_right));

    /**
     * This section is creating a custom MPI type to allow us to send non-contiguous
     * data in an easy manner.
     *
     * I.e., for sending column data to the left and right.
     */

    MPI_Datatype MPI_Column_type;

    MPI_Type_vector(n_rows,      // Number of elements in a column
                    1,           // one element at a time
                    n_cols + 2,  // the stride, distance between two elements
                    MPI_INT, &MPI_Column_type);

    MPI_Type_commit(&MPI_Column_type);

    /**
     * Now we start sending the data border cells to the adjacent halos.
     *
     * Each send and receive is tagged with an MPI_Request derived from the location
     * the data is being sent to. E.g., for a send from the top border of the current
     * rank to the bottom halo of the rank above, we have
     *     MPI_Request send_request_top, recv_request_top;
     */

    // Top border of current rank -> bottom halo of rank above
    MPI_Request send_request_top, recv_request_top;

    MPI_Isend(&grid(0, 0), n_cols, MPI_INT, up, 0, cartesian2d, &send_request_top);
    MPI_Irecv(&grid(n_rows, 0), n_cols, MPI_INT, down, 0, cartesian2d,
              &recv_request_top);

    // Bottom border of current rank -> top halo of rank below
    MPI_Request send_request_bottom, recv_request_bottom;

    MPI_Isend(&grid(n_rows - 1, 0), n_cols, MPI_INT, down, 1, cartesian2d,
              &send_request_bottom);
    MPI_Irecv(&grid(-1, 0), n_cols, MPI_INT, up, 1, cartesian2d, &recv_request_bottom);

    // Left border of current rank -> right halo of rank to the left
    MPI_Request send_request_left, recv_request_left;

    MPI_Isend(&grid(0, 0), 1, MPI_Column_type, left, 2, cartesian2d,
              &send_request_left);
    MPI_Irecv(&grid(0, n_cols), 1, MPI_Column_type, right, 2, cartesian2d,
              &recv_request_left);

    // Right border of current rank -> left halo of rank to the right
    MPI_Request send_request_right, recv_request_right;

    MPI_Isend(&grid(0, n_cols - 1), 1, MPI_Column_type, right, 3, cartesian2d,
              &send_request_right);
    MPI_Irecv(&grid(0, -1), 1, MPI_Column_type, left, 3, cartesian2d,
              &recv_request_right);

    // Top left cell of current rank -> bottom right halo of the rank to the top left
    MPI_Request send_request_top_left, recv_request_top_left;

    MPI_Isend(&grid(0, 0), 1, MPI_INT, top_left, 4, cartesian2d,
              &send_request_top_left);
    MPI_Irecv(&grid(n_rows, n_cols), 1, MPI_INT, bottom_right, 4, cartesian2d,
              &recv_request_top_left);

    // Top right cell of current rank -> bottom left halo of the rank to the top right
    MPI_Request send_request_top_right, recv_request_top_right;

    MPI_Isend(&grid(0, n_cols - 1), 1, MPI_INT, top_right, 5, cartesian2d,
              &send_request_top_right);
    MPI_Irecv(&grid(n_rows, -1), 1, MPI_INT, bottom_left, 5, cartesian2d,
              &recv_request_top_right);

    // Bottom left cell of current rank -> top right halo of the rank to the bottom left
    MPI_Request send_request_bottom_left, recv_request_bottom_left;

    MPI_Isend(&grid(n_rows - 1, 0), 1, MPI_INT, bottom_left, 6, cartesian2d,
              &send_request_bottom_left);
    MPI_Irecv(&grid(-1, n_cols), 1, MPI_INT, top_right, 6, cartesian2d,
              &recv_request_bottom_left);

    // Bottom right cell of current rank -> top left halo of the rank to the bottom
    // right
    MPI_Request send_request_bottom_right, recv_request_bottom_right;

    MPI_Isend(&grid(n_rows - 1, n_cols - 1), 1, MPI_INT, bottom_right, 7, cartesian2d,
              &send_request_bottom_right);
    MPI_Irecv(&grid(-1, -1), 1, MPI_INT, top_left, 7, cartesian2d,
              &recv_request_bottom_right);

    // Now we wait for the messages to all send and be recieved
    MPI_Wait(&send_request_top, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_top, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_bottom, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_bottom, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_left, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_left, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_right, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_right, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_top_left, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_top_left, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_top_right, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_top_right, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_bottom_left, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_bottom_left, MPI_STATUS_IGNORE);

    MPI_Wait(&send_request_bottom_right, MPI_STATUS_IGNORE);
    MPI_Wait(&recv_request_bottom_right, MPI_STATUS_IGNORE);

    if (rank == 1) {
        std::cout << "Rank: " << rank << std::endl;
        std::cout << "n_rows: " << n_rows << " n_cols: " << n_cols << std::endl;

        // print out bottom border
        for (int i = 0; i < n_cols; i++) {
            std::cout << grid(n_rows - 1, i) << " ";
        }

        std::cout << std::endl;
        // top halo
        for (int i = 0; i < n_cols; i++) {
            std::cout << grid(-1, i) << " ";
        }

        std::cout << std::endl << std::endl;
        // top border
        for (int i = 0; i < n_cols; i++) {
            std::cout << grid(0, i) << " ";
        }

        std::cout << std::endl;
        // bottom halo
        for (int i = 0; i < n_cols; i++) {
            std::cout << grid(n_rows, i) << " ";
        }

        std::cout << std::endl << std::endl;
        // left halo
        for (int i = 0; i < n_rows; i++) {
            std::cout << grid(i, -1) << " ";
        }
        std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        // right border
        for (int i = 0; i < n_rows; i++) {
            std::cout << grid(i, n_cols - 1) << " ";
        }
    }

    std::cout << std::endl;

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        std::cout << grid(n_rows, n_cols) << " " << bottom_right << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 1) {
        std::cout << grid(0, 0) << " " << top_left << std::endl;
    }

    MPI_Finalize();

    return 0;
}