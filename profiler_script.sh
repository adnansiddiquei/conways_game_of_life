#!/bin/bash
# This script simply runs the simulation numerous times with a bunch of different config, and outputs the 
# time to a file so it can be plotted.

run_simulation() {
    OMP_NUM_THREADS=${1:-3} # Default to 3 if not specified
    MPI_PROCESSES=${2:-2}   # Default to 2 if not specified
    OUTPUT_FILE=${3:-"./bin/out.txt"} # Default output file if not specified
    GRID_SIZE=${4:-5000}    # Default grid size if not specified
    PROBABILITY=${5:-0.7}   # Default probability if not specified
    RANDOM_SEED=${6:-42}    # Default random seed if not specified
    GENERATIONS=${7:-5000}  # Default generations if not specified
    VERBOSE=${8:-1}         # Default verbose level if not specified

    # Construct the command
    CMD="OMP_NUM_THREADS=$OMP_NUM_THREADS mpirun -np $MPI_PROCESSES ./bin/main --output $OUTPUT_FILE --grid-size $GRID_SIZE --probability $PROBABILITY --random-seed $RANDOM_SEED --generations $GENERATIONS --verbose $VERBOSE"

    # Execute the command
    eval $CMD
}

for iters in 0 1; do
    for GS in 2500 5000; do
        for ranks in 1 2 4 6 8 10; do
            for threads in 1 2 4 6 8 10; do
                run_simulation $ranks $threads "./bin/out.txt" $GS 0.7 21 200 1 >> ./src/plotting/mpi_omp.csv
            done
        done
    done
done