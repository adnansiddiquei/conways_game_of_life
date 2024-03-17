#include <gtest/gtest.h>
#include <mpi.h>

#include <array>

#include "conway.h"

class MPICommsTest : public ::testing::Test {
   protected:
    static void SetUpTestCase() { MPI_Init(nullptr, nullptr); }

    static void TearDownTestCase() { MPI_Finalize(); }

    // Helper function to set up Cartesian topology and return communicator
    MPI_Comm SetUpCartesianCommunicator(int n_rows, int n_cols,
                                        std::array<int, 2> &dims) {
        MPI_Comm cartesian2d;
        std::array<int, 2> periods = {1, 1};  // periodic boundary conditions
        int reorder = 1;
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims.data(), periods.data(), reorder,
                        &cartesian2d);
        return cartesian2d;
    }

    void SendRecvWaitAll(conway::ConwaysArray2DWithHalo &grid, MPI_Comm &cartesian2d,
                         std::array<int, 8> &neighbour_ranks) {
        int n_rows = grid.get_rows();
        int n_cols = grid.get_cols();

        MPI_Datatype MPI_Column_type;
        MPI_Type_vector(n_rows, 1, n_cols + 2, MPI_INT, &MPI_Column_type);
        MPI_Type_commit(&MPI_Column_type);

        std::array<MPI_Request, 8> send_requests;
        std::array<MPI_Request, 8> recv_requests;

        grid.MPI_Isend_all(cartesian2d, send_requests, neighbour_ranks,
                           MPI_Column_type);
        grid.MPI_Irecv_all(cartesian2d, recv_requests, neighbour_ranks,
                           MPI_Column_type);

        grid.MPI_Wait_all(send_requests, recv_requests);
    }

    void SetUpTestableGridBorderValues(conway::ConwaysArray2DWithHalo &grid, int rank) {
        int n_rows = grid.get_rows();
        int n_cols = grid.get_cols();

        // Give some border cells specific values which can be tested on the receiving
        // grid
        grid(0, 0) = (rank + 1) * 1;
        grid(0, 50) = (rank + 1) * 2;
        grid(0, n_cols - 1) = (rank + 1) * 3;
        grid(10, n_cols - 1) = (rank + 1) * 4;
        grid(n_rows - 1, n_cols - 1) = (rank + 1) * 5;
        grid(n_rows - 1, 50) = (rank + 1) * 6;
        grid(n_rows - 1, 0) = (rank + 1) * 7;
        grid(10, 0) = (rank + 1) * 8;
    }

    void TestHaloValueHaveReceivedCorrectly(conway::ConwaysArray2DWithHalo &grid,
                                            std::array<int, 8> &neighbour_ranks) {
        int n_rows = grid.get_rows();
        int n_cols = grid.get_cols();

        ASSERT_EQ(grid(n_rows, n_cols), (neighbour_ranks[4] + 1) * 1);
        ASSERT_EQ(grid(n_rows, 50), (neighbour_ranks[5] + 1) * 2);
        ASSERT_EQ(grid(n_rows, -1), (neighbour_ranks[6] + 1) * 3);
        ASSERT_EQ(grid(10, -1), (neighbour_ranks[7] + 1) * 4);
        ASSERT_EQ(grid(-1, -1), (neighbour_ranks[0] + 1) * 5);
        ASSERT_EQ(grid(-1, 50), (neighbour_ranks[1] + 1) * 6);
        ASSERT_EQ(grid(-1, n_cols), (neighbour_ranks[2] + 1) * 7);
        ASSERT_EQ(grid(10, n_cols), (neighbour_ranks[3] + 1) * 8);
    }
};

TEST_F(MPICommsTest, row_wise_decomposition) {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    std::array<int, 2> dims = {6, 1};  // Testing a row-wise decompisition
    int n_rows = 20, n_cols = 100;
    conway::ConwaysArray2DWithHalo grid(n_rows, n_cols);
    grid.fill_randomly(0.7, -1, true);  // Initialize the grid with some data

    MPI_Comm cartesian2d = SetUpCartesianCommunicator(n_rows, n_cols, dims);
    SetUpTestableGridBorderValues(grid, rank);
    std::array<int, 8> neighbour_ranks = grid.get_neighbour_ranks(cartesian2d, dims);
    SendRecvWaitAll(
        grid, cartesian2d,
        neighbour_ranks);  // send, recv and wait until all the data has been received
    TestHaloValueHaveReceivedCorrectly(grid,
                                       neighbour_ranks);  // test data was received okay
}

TEST_F(MPICommsTest, column_wise_decomposition) {
    int rank, n_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);

    std::array<int, 2> dims = {1, 6};  // Testing a column-wise decompisition
    int n_rows = 20, n_cols = 100;
    conway::ConwaysArray2DWithHalo grid(n_rows, n_cols);
    grid.fill_randomly(0.7, -1, true);  // Initialize the grid with some data

    MPI_Comm cartesian2d = SetUpCartesianCommunicator(n_rows, n_cols, dims);
    SetUpTestableGridBorderValues(grid, rank);
    std::array<int, 8> neighbour_ranks = grid.get_neighbour_ranks(cartesian2d, dims);
    SendRecvWaitAll(
        grid, cartesian2d,
        neighbour_ranks);  // send, recv and wait until all the data has been received
    TestHaloValueHaveReceivedCorrectly(grid,
                                       neighbour_ranks);  // test data was received okay
}
