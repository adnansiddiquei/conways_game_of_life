#include "conway.h"

#include <omp.h>

#include <random>

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