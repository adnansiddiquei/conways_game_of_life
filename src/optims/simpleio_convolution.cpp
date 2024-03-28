#include <omp.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "array2d.h"
#include "conway.h"
#include "timer.h"

/**
 * @brief  Times the SimpleIO neighbour counting method for different OMP_NUM_THREADS
 * and outputs a csv of results to `src/plotting`
 */
int main(int argc, char *argv[]) {
    // Open the output file
    std::ofstream output_file("src/plotting/simpleio_convolution.csv");

    // Throw an error in case the file failed to open
    if (!output_file.is_open()) {
        std::cerr << "Unable to open output file.";
        std::exit(1);
    }

    // Compute all of the grid_sizes that we will time on
    std::vector<int> grid_sizes = [&]() {
        std::vector<int> gs;
        gs.reserve(4);

        gs.push_back(5000);
        gs.push_back(10000);
        gs.push_back(15000);
        gs.push_back(20000);

        return gs;
    }();

    // Compute all of the OMP_NUM_THREADS we want to work on
    std::vector<int> all_num_threads = [&]() {
        std::vector<int> ant;
        ant.reserve(10);

        for (int i = 1; i < 11; i++) {
            ant.push_back(i);
        }

        return ant;
    }();

    output_file << "method,num_threads,grid_size,duration" << std::endl;

    for (int iters = 0; iters < 3; iters++) {
        // Time the SimpleIO convolution
        for (int num_threads : all_num_threads) {
            for (int grid_size : grid_sizes) {
                omp_set_num_threads(num_threads);
                num_threads = omp_get_max_threads();

                conway::ConwaysArray2DWithHalo grid(grid_size, grid_size);
                grid.fill_randomly(0.5, -1, true);  // fill all cells, including halo

                array2d::Array2D<int> neighbour_count(grid_size, grid_size, 0);

                timer::start_clock();

                // do the neighbour count
                grid.simple_convolve_inner(neighbour_count);
                grid.simple_convolve_outer(neighbour_count);

                double duration = timer::get_split();

                output_file << "simple_inner_outer," << num_threads << "," << grid_size
                            << "," << duration << std::endl;
            }
        }
    }

    output_file.close();
}