#include <mpi.h>
#include <omp.h>

#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <utility>

#include "array2d.h"
#include "conway.h"
#include "timer.h"

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
    /**
     * Parse all of the arguments passed into the script.
     * See the README.md for a complete overview of all of the arguments
     */
    std::string input_filepath = parse_arg(argc, argv, "--input", false);
    std::string output_filepath = parse_arg(argc, argv, "--output", false);
    int verbose = parse_arg_int(argc, argv, "--verbose", false);
    int grid_size = parse_arg_int(argc, argv, "--grid-size", true);
    int generations = parse_arg_int(argc, argv, "--generations", true);
    int random_seed;
    float probability;

    // If --input was not provided then we need the following params to generate the
    // grid
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

    // periodic boundary conditions in both axes
    std::array<int, 2> periods = {1, 1};
    int reorder = 1;

    // create the cartesian2d communicator
    MPI_Comm cartesian2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), reorder,
                    &cartesian2d);

    // Get the grid size of the decomposed grid
    std::array<int, 2> decomposed_grid_size =
        conway::get_decomposed_grid_size(rank, n_ranks, grid_size, decomposition_type);

    int n_rows = decomposed_grid_size[0];
    int n_cols = decomposed_grid_size[1];

    // Create the 2D array that will represent the decomposed domain
    conway::ConwaysArray2DWithHalo grid(n_rows, n_cols);
    conway::ConwaysArray2DWithHalo *large_grid;

    /**
     * Compute the number of rows per rank and exactly how many rows the last rank
     * will have. Because if the grid_size is not evenly divisible by the number of
     * ranks, then the last rank will take all of the remaining ranks.
     */
    int rows_per_rank = grid_size / n_ranks;
    int extra_rows = grid_size % n_ranks;
    int rows_for_last_rank = rows_per_rank + extra_rows;

    // Now generate all of the data on rank 0
    if (rank == 0) {
        large_grid = new conway::ConwaysArray2DWithHalo(grid_size, grid_size);

        if (input_filepath == "") {
            // fill the grid randomly if no input file was provided
            large_grid->fill_randomly(probability, random_seed);
        } else {
            // otherwise read from the text file provided
            conway::read_from_text_file(*large_grid, input_filepath);
        }
    }

    /**
     * Here we define two custom MPI_Datatype variables, which make it easy to split the
     * data from rank 0 to every other rank, and later collect the data back into rank
     * 0.
     *
     * MPI_Block_type_1 defines a type which represents a block of data corresponding
     * to the grid size for every rank except the last one. Note the `n_cols + 2`,
     * this is required because we ConwaysArray2DWithHalo contains a 1 cell halo, it is
     * not simply a (rows_per_rank * n_cols) block of data as we want to ignore the
     * data in the halos.
     *
     * MPI_Block_type_2 is similar, it just accounts for the fact that the last rank
     * may have a different number of rows.
     */
    MPI_Datatype MPI_Block_type_1;
    MPI_Type_vector(rows_per_rank, n_cols, n_cols + 2, MPI_INT, &MPI_Block_type_1);
    MPI_Type_commit(&MPI_Block_type_1);

    MPI_Datatype MPI_Block_type_2;
    MPI_Type_vector(rows_for_last_rank, n_cols, n_cols + 2, MPI_INT, &MPI_Block_type_2);
    MPI_Type_commit(&MPI_Block_type_2);

    // If rank 0, then we send the corresponding data to the relevant rank
    if (rank == 0) {
        MPI_Request null_req;

        for (int i = 0; i < n_ranks; i++) {
            // This MPI_Isend is a bit convoluted with the ternary operator, but it
            // just ensures that if we are sending data to the last rank, we use
            // MPI_Block_type_2
            MPI_Isend(&(*large_grid)(i * rows_per_rank, 0), 1,
                      i != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2, i, i,
                      cartesian2d, &null_req);
        }
    }

    // Recieve all of the data into each rank
    MPI_Recv(&grid(0, 0), 1, rank != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2,
             0, rank, cartesian2d, MPI_STATUS_IGNORE);

    // Free the memory by deleting the large grid
    if (rank == 0) {
        delete large_grid;
    }

    /**
     * MPI_Column_type serves a similar function to MPI_Block_type_1.
     *
     * It defines a data type that spans each of the left and right borders.
     * With this, we can easily send and recieve the left and right borders into
     * the left and right halos.
     *
     * Whilst this is not strictly necessary (given that we are only doing row
     * decomp and technically don't need to send left and right borders as they live
     * on the same rank), it is implemented for generalisability.
     */
    MPI_Datatype MPI_Column_type;

    MPI_Type_vector(n_rows,      // Number of elements in a column
                    1,           // one element at a time
                    n_cols + 2,  // the stride, distance between two elements
                    MPI_INT, &MPI_Column_type);

    MPI_Type_commit(&MPI_Column_type);

    /**
     * This gets the neighbours of the current rank.
     *
     * Each rank has 8 neighbours:
     *      top-left, up, top-right, right, bottom-right, bottom, bottom-left, left
     *
     * These corresponding ranks are saved into this array. E.g., the 3rd element
     * of neighbour_ranks will be the rank that is to the top right of the current
     * rank in the cartesian2d topology.
     */
    std::array<int, 8> neighbour_ranks = grid.get_neighbour_ranks(cartesian2d, dims);

    // Evolve the grid for the specified number of generations
    timer::start_clock();
    for (int i = 0; i < generations; i++) {
        /**
         * Implement the SimpleIO method (as described in the report).
         *
         * We count the neighbours of the inner cells that are not adjacent to halos
         * while we wait for the halo exchange to complete. Then we count the
         * neighbours for the outer cells.
         */

        std::array<MPI_Request, 8> send_requests;
        std::array<MPI_Request, 8> recv_requests;

        // Send all border cells to the corresponding neighbours, non-blocking
        grid.MPI_Isend_all(cartesian2d, send_requests, neighbour_ranks,
                           MPI_Column_type);
        // Recieve all the border cells into the correct locations, non-blocking
        grid.MPI_Irecv_all(cartesian2d, recv_requests, neighbour_ranks,
                           MPI_Column_type);

        // Start counting the neighbours of the inner cells.
        array2d::Array2D<int> neighbour_count(n_rows, n_cols);
        grid.simple_convolve_inner(neighbour_count);

        // Now we need to wait for the halo exchange to finish.
        grid.MPI_Wait_all(send_requests, recv_requests);

        // Count the outer neighbours
        grid.simple_convolve_outer(neighbour_count);

        // Transition to the next state
        grid.transition_lookup(neighbour_count);
    }

    // If the user specified `--verbose 1` when running the script, then we want to
    // output some statistics on the above simulation that can be used for performance
    // testing.
    if (rank == 0 && verbose == 1) {
        double duration = timer::get_split();
        std::cout << n_ranks << "," << omp_get_max_threads() << "," << grid_size << ","
                  << generations << "," << duration << std::endl;
    }

    /**
     * Now we collate the entire grid back onto rank 0 to save it down into a text
     * file.
     */

    conway::ConwaysArray2DWithHalo *final_grid;

    if (rank == 0) {
        final_grid = new conway::ConwaysArray2DWithHalo(grid_size, grid_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Request null_req;
    std::vector<MPI_Request> final_send_requests(n_ranks, MPI_REQUEST_NULL);

    // Loop over every single rank and send their data to rank 0
    for (int i = 0; i < n_ranks; i++) {
        if (i != rank) {
            continue;
        }

        // The ternary operator serves the same purpose as before
        MPI_Isend(&grid(0, 0), 1,
                  i != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2, 0, i,
                  cartesian2d, &final_send_requests[i]);
    }

    std::vector<MPI_Request> final_recv_requests(n_ranks, MPI_REQUEST_NULL);

    // Now receive all the data into rank 0
    if (rank == 0) {
        for (int i = 0; i < n_ranks; i++) {
            MPI_Irecv(&(*final_grid)(i * rows_per_rank, 0), 1,
                      i != n_ranks - 1 ? MPI_Block_type_1 : MPI_Block_type_2, i, i,
                      cartesian2d, &final_recv_requests[i]);
        }
    }

    // Wait for final send requests to complete
    MPI_Wait(&final_send_requests[rank], MPI_STATUS_IGNORE);

    // If rank 0, also wait for all receive operations to complete before proceeding
    if (rank == 0) {
        MPI_Waitall(n_ranks, final_recv_requests.data(), MPI_STATUSES_IGNORE);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // Save the grid to a file
    if (rank == 0) {
        conway::save_to_text_file(*final_grid, output_filepath);
        delete final_grid;
    }

    // Free all of the custom MPI types
    MPI_Type_free(&MPI_Block_type_1);
    MPI_Type_free(&MPI_Block_type_2);
    MPI_Type_free(&MPI_Column_type);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

    return 0;
}