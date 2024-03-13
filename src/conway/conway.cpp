#include "conway.h"

#include <random>

#include "array2d.h"

using namespace conway;
using namespace array2d;

ConwaysArray2DWithHalo::ConwaysArray2DWithHalo(int n_rows, int n_cols)
    : Array2DWithHalo<int>(n_rows, n_cols){
          // Nothing required
      };

void ConwaysArray2DWithHalo::fill_randomly(float probability, int random_seed = -1) {
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

    // Fill in every cell value (excluding halo cells) with a 0 or 1, according to
    // `probability`
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            (*this)(i, j) = dis(gen);
        }
    }
}