#!/bin/bash
# If you have a whole bunch of input files and you want to run them all just change all of the below
# so they all point to the correct input file. Change the --output as well so they all save to different locations
# and don't overwrite each other. Make sure to also change --grid-size so it exactly matches the grid size of the input file.
# --generations is how many iterations you want to evolve the simulation for.
mpirun -np 4 bin/main --input ./tests/test_grid_glider.txt --output ./bin/out1.txt --grid-size 81 --generations 312
mpirun -np 4 bin/main --input ./tests/test_grid_glider.txt --output ./bin/out1.txt --grid-size 81 --generations 312
mpirun -np 4 bin/main --input ./tests/test_grid_glider.txt --output ./bin/out1.txt --grid-size 81 --generations 312
mpirun -np 4 bin/main --input ./tests/test_grid_glider.txt --output ./bin/out1.txt --grid-size 81 --generations 312

# If you just want to have some fun, this will randomly generate a 150x150 grid with a 70% chance of a cell being alive
# and run it for 500 generations. It will save the output to bin/out2.txt
mpirun -np 4 bin/main --output ./bin/out2.txt --grid-size 150 --probability 0.7 --random-seed 21 --generations 500

# And then submit this script from the root folder of this repo with:
# sbatch scripts/slurm_submit_runner.peta4-icelake

# Or alternatively you can just run this script directly from the root folder, on your local machine, with
# ./scripts/runner.sh