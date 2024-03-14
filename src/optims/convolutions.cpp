#include <iostream>

#include "array2d.h"
#include "conway.h"
#include "timer.h"

int main(int argc, char *argv[]) {
    for (int i = 100; i < 1000; i += 100) {
        conway::ConwaysArray2DWithHalo grid(i, i);
        grid.fill_randomly(0.5, -1, true);  // fill all cells, including halo

        array2d::Array2D<int> neighbour_count(i, i, 0);

        timer::start_clock();

        grid.simple_convolve(neighbour_count);
        
        double duration = timer::get_split();
        std::cout << duration << std::endl;
    }
}