#include <mpi.h>

#include <array>

#include "array2d.h"

#ifndef AS3438_CONWAY_H
#define AS3438_CONWAY_H

// TODO: comment
namespace conway {
class ConwaysArray2DWithHalo : public array2d::Array2DWithHalo<int> {
   public:
    ConwaysArray2DWithHalo(int n_rows, int n_cols);

    void fill_randomly(float probability, int random_seed = -1, bool fill_halo = false);

    void simple_convolve(array2d::Array2D<int> &neighbour_count);

    void separable_convolution(array2d::Array2D<int> &neighbour_count);

    void simple_convolve_inner(array2d::Array2D<int> &neighbour_count);

    void simple_convolve_outer(array2d::Array2D<int> &neighbour_count);

    void transition_ifs(array2d::Array2D<int> &neighbour_count);

    void transition_lookup(array2d::Array2D<int> &neighbour_count);

    void transition_bitwise(array2d::Array2D<int> &neighbour_count);

    void MPI_Isend_all(MPI_Comm &cartesian2d, std::array<MPI_Request, 8> &requests,
                       std::array<int, 8> &neighbours, MPI_Datatype &MPI_Column_type);

    void MPI_Irecv_all(MPI_Comm &cartesian2d, std::array<MPI_Request, 8> &requests,
                       std::array<int, 8> &neighbours, MPI_Datatype &MPI_Column_type);

    void MPI_Wait_all(std::array<MPI_Request, 8> &send_reqs,
                      std::array<MPI_Request, 8> &recv_reqs);
};
}  // namespace conway

#endif  // AS3438_CONWAY_H
