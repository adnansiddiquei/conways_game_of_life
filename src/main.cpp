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

int main(int argc, char *argv[]) {
    // parse the arguments passed into the script
    int grid_size = parse_arg_int(argc, argv, "--grid-size", true);
    int random_seed = parse_arg_int(argc, argv, "--random-seed", false);
    float probability = parse_arg_float(argc, argv, "--probability", true);

    // Now we create the MPI ranks and set up the cartesian communicator
    MPI_Init(&argc, &argv);

    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    // settings for the cartesian2d communicator
    int dims[2] = {1,
                   n_ranks};  // column-wise decomposition: 1 row and `n_ranks` columns
    int periods[2] = {1, 1};  // periodic boundary conditions in both axes
    int reorder = 1;

    // create the cartesian2d communicator
    MPI_Comm cartesian2d;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cartesian2d);

    // now save down which ranks are in each direction of each rank
    int left, right, up, down;
    MPI_Cart_shift(cartesian2d, 1, 1, &left, &right);
    MPI_Cart_shift(cartesian2d, 0, 1, &down, &up);

    // Get the grid size of the decomposed grid
    /**
     * TODO: In the case that grid_size / n_ranks == 0.
     * This code needs to be edited to handle when the grid might be very small.
     * Maybe in that case just don't even bother doing domain decomposition.
     */
    std::array<int, 2> decomposed_grid_size = [&rank, &n_ranks,
                                               &grid_size]() -> std::array<int, 2> {
        if (rank + 1 != n_ranks) {
            return {grid_size, grid_size / n_ranks};
        } else {
            // for the last rank, the number of columns may be different so the below
            // line accounts for the fact that grid_size may not be perfectly divisible
            // by n_ranks
            return {grid_size, grid_size - (n_ranks - 1) * (grid_size / n_ranks)};
        }
    }();

    // Create the 2D array that will represent the decomposed domain
    conway::ConwaysArray2DWithHalo grid(decomposed_grid_size[0], decomposed_grid_size[1]);

    // Fill the grid (excluding the halo) with 1s and 0s according to the probability
    // and random_seed the user inputted on the command line.
    grid.fill_randomly(probability, random_seed);

    MPI_Finalize();

    return 0;
}