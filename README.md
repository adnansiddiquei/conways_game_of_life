# C2: Research Computing (CW Assignment) - Conway's Game of Life
## Table of Contents
1. [Run](#run)
2. [Generating the Figures / Performance Testing](#pt)
3. [Tests](#tests)
4. [Documentation](#docs)
5. [Report](#report)

## <a name="run"></a> 1. Running the simulation
Clone the repository and then run
```bash
make all
```
in your terminal, within the root of this repo. This will build the C++ targets and place them into the `bin/` folder.

The simulation is run using the `main` executable in the `bin/` folder, and by passing it the appropriate arguments.

### 1.1. Run locally
This section explains how to run the simulation on your laptop.

The below two examples are valid examples that can be run now.
```bash
# this will run a 81x81 simulation across 5 ranks, with row decomposition, over 312 generations on the input file provided.
mpirun -np 5 bin/main --input ./tests/test_grid_glider.txt --output ./bin/out1.txt --grid-size 81 --generations 312
```
```bash
# this will randomly generate a 150x150 grid, scatter to 6 ranks and run for 500 generations.
mpirun -np 6 bin/main --output ./bin/out2.txt --grid-size 150 --probability 0.7 --random-seed 21 --generations 500
```
Full usage documenation below:
```bash
# Usage:
#  $ mpirun -np [num_ranks] ./bin/main --file [arg] --grid-size [arg] --generations [arg]
#  $ mpirun -np [num_ranks] ./bin/main --grid-size [arg] --generations [arg] --probability [arg] [--random--seed [arg]]
#
# * --generations: (int) required. The number of generations to evolve the grid for.
#
# * --grid-size: (int) required. The dimensions for each square side of the entire
#                 simulation. This must be provided in even though the grid-size
#                 can technically be inferred from the file.
#
# * --output: (string) required. File path to save output to.

# * --input: (string) required if --probability is not passed in. This is the file 
#           path to a .txt file containing the initial state.
#
# * --probability: (float, [0, 1)) required if --file is not passed in. This is 
#                 the probability of a cell being alive in the initial state when 
#                 the initial state is randomly generated.
#
# * --random-seed: (string) optional. Can be passed in with --probability to 
#                 make the simulation reproducible.
#
# * --verbose: (int) if provided, and if set to `1`, then the simulation will print the
#             a verbose output at the end of the simulation indicating (in those order)
#             `num_ranks,OMP_NUM_THREADS,grid_size,generations,simulation_duration`. This
#             was mainly used for performance testing and can be ignored for general use.
```

If providing an input file with `--input`, the file must be in a format identical to [`tests/test_grid_glider.txt`](tests/test_grid_glider.txt).
The file must be a square grid (N lines, and N characters per line), with no delimiters. A `1` represents a living cell and a `0`
represents a dead cell. The `--grid-size` parameter must also be provided and it must match exactly with
the number of lines and characters per line in the input file.

### 1.2. Run on the HPC
1. SSH into the HPC
2. Clone the repo
3. Run `nice -19 make all` to build the code.
4. Edit the `scripts/runner.sh` template script to include the desired tests or simulations you would like to run.
5. Submit the job to the HPC using the `scripts/slurm_submit_runner.peta4-icelake` script. NOTE - please submit this from
from the root of the repo using `sbatch scripts/slurm_submit_runner.peta4-icelake` to ensure that `$workdir` is set up correctly
in slurm. When editing the `runner.sh` script, ensure that all file paths (`--input` and `--output` args) are relative from the root of the repo as well.

## <a name="pt"></a> 2. Generating the Figures / Performance Testing
The `src/plotting` folder contains a `plots.ipynb` which can be used to generate the plots used in the report. To run this
code you will first need to activate the conda environment provided in the `environment.yml` file and run the required scripts
to generate the data.

To activate the conda environment, run:
```bash
make conda-env
````
```bash
conda activate as3438_c2cw
```

### 2.1. Figures 2 and 3
You will note a `src/optims` folder, this folder contains a bunch of additional executables that can be run to generate
the data for Figures 2 and 3. These can be run with 
```bash
./bin/optims_convolutions
./bin/optims_simpleio_convolution
./bin/optims_transitions
./bin/optims_lookup_transition
```
See the relevant source file in `src/optims` or the report itself for an explanation on the data that these executables generate.
Note that all of these scripts will output a csv directly to the `src/plotting` folder.

The performance on your local machine will also differ to those in the report, ensure that the machine you are running on
can support up to 10 OMP threads.

### 2.1. Figures 5 and 6
These figures are the full simulation performance tests that were run on the Icelake partition of the Cambridge HPC, on
a single 76 core node. The scripts for these live in the `scripts/` folder: `profiler_script_icelake_1.sh` and 
`profiler_script_icelake_2.sh`. If you inspect these scripts you will note that they are simply running the `main` executable
with different parameters with the `--verbose 1` argument, and piping the output to a csv file.

Please submit these scripts to the HPC using (note that the second script takes a few hours to run)
```bash
sbatch scripts/slurm_submit_1.peta4-icelake
sbatch scripts/slurm_submit_2.peta4-icelake
```
Same as before, the outputs will be placed into the `src/plotting` folder. To plot these, `scp` them back into the same folder
on your local computer and run the corresponding cells in the `plots.ipynb` notebook.

## <a name="tests"></a> 3. Tests
After running `make all` you can run `make test` and this will run the entire GoogleTest testing suite within the `tests/` folder.

Ensure that your local machine can handle generating up to 6 MPI ranks, as the tests in `tests/test_mpi_comms.cpp`
attempts to run a small simulation across 6 ranks.

## <a name="docs"></a> 4. Documentation
The documentation is generated by doxygen, run:
```bash
make docs
```
And then open `docs/html/index.html` in your browser.


## <a name="report"></a> 4. Report
The report is located at [report/out/main.pdf](report/out/main.pdf).
