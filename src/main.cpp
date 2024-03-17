#include <mpi.h>

#include <array>
#include <cstdlib>
#include <fstream>
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

int main(int argc, char *argv[]) {
    // parse the arguments passed into the script
    std::string input_filepath = parse_arg(argc, argv, "--input", false);
    std::string output_filepath = parse_arg(argc, argv, "--output", false);
    int grid_size = parse_arg_int(argc, argv, "--grid-size", true);
    int generations = parse_arg_int(argc, argv, "--generations", true);
    int random_seed;
    float probability;

    // If a file was not provided then we need the following params
    if (input_filepath == "") {
        random_seed = parse_arg_int(argc, argv, "--random-seed", false);
        probability = parse_arg_float(argc, argv, "--probability", true);
    }

    std::string decomposition_type = "row";

    // ---- COLUMN DECOMP HAS BEEN REMOVED FOR THE MOMENT ----
    // std::string decomposition_type = parse_arg(argc, argv, "--mpi-decomp", true);
    // Ensure that a valid decomposition type was specified
    // if (decomposition_type != "column" & decomposition_type != "row" &
    //     decomposition_type != "grid") {
    //     std::cerr << "Out of range: --mpi-decomp must be one of \"column\", \"row\"
    //     or "
    //                  "\"grid\"."
    //               << std::endl;
    //     std::exit(1);
    // }

    // Ensure a non-negative number of generations was passed in
    if (generations < 0) {
        std::cerr << "Out of range: --generations must non-negative." << std::endl;
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
    // TODO: What happens when the grid size is really small?
    std::array<int, 2> decomposed_grid_size =
        conway::get_decomposed_grid_size(rank, n_ranks, grid_size, decomposition_type);

    int n_rows = decomposed_grid_size[0];
    int n_cols = decomposed_grid_size[1];

    // Create the 2D array that will represent the decomposed domain
    conway::ConwaysArray2DWithHalo grid(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo *large_grid;

    int rows_per_rank = grid_size / n_ranks;
    int extra_rows = grid_size % n_ranks;
    int rows_for_last_rank = rows_per_rank + extra_rows;

    if (rank == 0) {
        large_grid = new conway::ConwaysArray2DWithHalo(grid_size, grid_size);

        if (input_filepath == "") {
            large_grid->fill_randomly(probability, random_seed);
        } else {
            conway::read_from_text_file(*large_grid, input_filepath);
        }
    }

    MPI_Datatype MPI_Block_type_1;
    MPI_Type_vector(rows_per_rank, n_cols, n_cols + 2, MPI_INT, &MPI_Block_type_1);
    MPI_Type_commit(&MPI_Block_type_1);

    MPI_Datatype MPI_Block_type_2;
    MPI_Type_vector(rows_for_last_rank, n_cols, n_cols + 2, MPI_INT, &MPI_Block_type_2);
    MPI_Type_commit(&MPI_Block_type_2);

    if (rank == 0) {
        MPI_Request null_req;

        for (int i = 0; i < n_ranks; i++) {
            MPI_Isend(&(*large_grid)(i * rows_per_rank, 0), 1,
                      i != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2, i, i,
                      cartesian2d, &null_req);
        }
    }

    MPI_Recv(&grid(0, 0), 1, rank != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2,
             0, rank, cartesian2d, MPI_STATUS_IGNORE);

    if (rank == 0) {
        delete large_grid;
    }

    // Create column data type so we can easily send and receive column data
    MPI_Datatype MPI_Column_type;

    MPI_Type_vector(n_rows,      // Number of elements in a column
                    1,           // one element at a time
                    n_cols + 2,  // the stride, distance between two elements
                    MPI_INT, &MPI_Column_type);

    MPI_Type_commit(&MPI_Column_type);

    // get the ranks of the neighbours
    std::array<int, 8> neighbour_ranks = grid.get_neighbour_ranks(cartesian2d, dims);

    // Evolve the grid for the specified number of generations
    for (int i = 0; i < generations; i++) {
        std::array<MPI_Request, 8> send_requests;
        std::array<MPI_Request, 8> recv_requests;

        // send and receieve all comms, non-blocking
        grid.MPI_Isend_all(cartesian2d, send_requests, neighbour_ranks,
                           MPI_Column_type);
        grid.MPI_Irecv_all(cartesian2d, recv_requests, neighbour_ranks,
                           MPI_Column_type);

        // Do the neighbour count on the inner portion while we wait for the comms to
        // send
        array2d::Array2D<int> neighbour_count(n_rows, n_cols);
        grid.simple_convolve_inner(neighbour_count);

        // Wait for the halos to arrive
        grid.MPI_Wait_all(send_requests, recv_requests);

        // Count the outer neighbours
        grid.simple_convolve_outer(neighbour_count);

        // Transition to the next state
        grid.transition_lookup(neighbour_count);
    }

    conway::ConwaysArray2DWithHalo *final_grid;

    if (rank == 0) {
        final_grid = new conway::ConwaysArray2DWithHalo(grid_size, grid_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Request null_req;

    for (int i = 0; i < n_ranks; i++) {
        MPI_Isend(&grid(0, 0), 1,
                  i != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2, 0, i,
                  cartesian2d, &null_req);
    }

    if (rank == 0) {
        for (int i = 0; i < n_ranks; i++) {
            MPI_Recv(&(*final_grid)(i * rows_per_rank, 0), 1,
                     i != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2, i, i,
                     cartesian2d, MPI_STATUS_IGNORE);
        }
    }

    if (rank == 0) {
        conway::save_to_text_file(*final_grid, output_filepath);
        delete final_grid;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Type_free(&MPI_Block_type_1);
    MPI_Type_free(&MPI_Block_type_2);
    MPI_Type_free(&MPI_Column_type);

    MPI_Finalize();

    return 0;
}