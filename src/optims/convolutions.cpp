#include <omp.h>

#include <fstream>
#include <iostream>
#include <vector>

#include "array2d.h"
#include "conway.h"
#include "timer.h"

int main(int argc, char *argv[]) {
    // Open the output file
    std::ofstream output_file("src/plotting/convolutions.csv");

    // Throw an error in case the file failed to open
    if (!output_file.is_open()) {
        std::cerr << "Unable to open output file.";
        std::exit(1);
    }

    // Compute all of the grid_sizes that we will time on
    std::vector<int> grid_sizes = [&]() {
        std::vector<int> gs;
        gs.reserve(10);

        for (int i = 1; i < 13; i++) {
            gs.push_back(i * 1000);
        }

        return gs;
    }();

    // Compute all of the OMP_NUM_THREADS we want to work on
    std::vector<int> all_num_threads = [&]() {
        std::vector<int> ant;
        ant.reserve(4);

        ant.push_back(1);
        ant.push_back(5);
        ant.push_back(10);

        return ant;
    }();

    output_file << "method,num_threads,grid_size,duration" << std::endl;

    // Time the simple_convolution
    for (int num_threads : all_num_threads) {
        for (int grid_size : grid_sizes) {
            omp_set_num_threads(num_threads);
            num_threads = omp_get_max_threads();

            conway::ConwaysArray2DWithHalo grid(grid_size, grid_size);
            grid.fill_randomly(0.5, -1, true);  // fill all cells, including halo

            array2d::Array2D<int> neighbour_count(grid_size, grid_size, 0);

            timer::start_clock();

            grid.simple_convolve(neighbour_count);  // do the neighbour count

            double duration = timer::get_split();

            output_file << "simple_convolve," << num_threads << "," << grid_size << ","
                        << duration / 1000 << std::endl;

            // int i = 3, j = 7;
            // int sum = grid(i, j - 1) + grid(i, j + 1) + grid(i - 1, j - 1) +
            //           grid(i - 1, j) + grid(i - 1, j + 1) + grid(i + 1, j - 1) +
            //           grid(i + 1, j) + grid(i + 1, j + 1);

            // std::cout << sum << " " << neighbour_count(i, j) << std::endl;
        }
    }

    // Time the separable_convolution
    for (int num_threads : all_num_threads) {
        for (int grid_size : grid_sizes) {
            omp_set_num_threads(num_threads);
            num_threads = omp_get_max_threads();

            conway::ConwaysArray2DWithHalo grid(grid_size, grid_size);
            grid.fill_randomly(0.5, -1, true);  // fill all cells, including halo

            array2d::Array2D<int> neighbour_count(grid_size, grid_size, 0);

            timer::start_clock();

            grid.separable_convolution(neighbour_count);  // do the neighbour count

            double duration = timer::get_split();

            output_file << "separable_convolution," << num_threads << "," << grid_size
                        << "," << duration / 1000 << std::endl;
        }
    }

    output_file.close();
}