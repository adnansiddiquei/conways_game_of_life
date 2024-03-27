#!/bin/bash
# This script runs the simulation multiple times on icelake, with differing MPI ranks and threads.

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

cores=76

out_file=./src/plotting/mpi_vs_omp_hpc.csv
err_file=./src/plotting/mpi_vs_omp_hpc_errors.txt

# add the header if the file does not exist yet
if [ ! -f "$out_file" ]; then
  echo "ranks,threads,grid_size,generations,duration" > "$out_file"
fi

for GS in 2500 5000; do  # for each grid size
    for i in 1 2 4 19 38 76; do
        for j in 1 2 4 19 38 76; do
            for iters in 0 1; do  # run the script for the pair of ranks and threads
                run_simulation $j $i "./bin/out.txt" $GS 0.7 21 200 1 2>> "$err_file" | grep '^[0-9]\+,[0-9]\+,.*' >> "$out_file"
                run_simulation $i $j "./bin/out.txt" $GS 0.7 21 200 1 2>> "$err_file" | grep '^[0-9]\+,[0-9]\+,.*' >> "$out_file"
            done
        done
    done
done