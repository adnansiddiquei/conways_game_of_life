# C2: Research Computing (CW Assignment) - Conway's Game of Life
## Table of Contents
1. [Run](#run)
2. [Generating the figures / performance testing](#pt)
3. [Understanding the code structure](#ucs)
4. [Tests](#tests)
5. [Documentation](#docs)
6. [Report](#report)
7. [Use of auto-generation tools](#auto)

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

## <a name="ucs"></a> 3. Understanding the Code Structure

The basic summary or how this codebase works is as follows
1. `tests/` contains all of the tests written with GTest.
2. `src/` contains all of the source code.
3. `bin/` contains all of the executables after `make all` is run.
4. `scripts/` contains all of the scripts used to run the performance tests on the HPC.
5. `report/` contains the LaTeX source code for the report.
6. `src/plotting` contains the Jupyter notebook used to generate the plots in the report. To run the code in this
folder, you will need to activate the conda environment provided in the `environment.yml` file.
7. `src/optims` folder contains additional executables used for performance testing.

Inside the `src/` folder, we have a few different modules.
 1. The first module of interest is the `array2d` module. This module implements the `Array2D` class
which is inherited by `Array2DWithHalo` (also implemented in this module) and `ConwaysArray2DWithHalo`. It is a simple 
implementation of a 2D array in row-major order, adapted from J.R. Fergusson's code.
2. The next module of interest is the `conway` module. This module implements a specific class that inherits
from `Array2DWithHalo` called `ConwaysArray2DWithHalo`. This class implements a lot of specific logic for Conway's Game 
of Life. The namespace `conway` (defined in this module) also implements a few other helper functions. Generally,
most logic that is tied specifically to Conway's Game of Life is implemented in this module.
3. `timer` implements a timer for performance testing. This module is entirely attributed to J.R. Fergusson and used
without any edits to the original code.

The `main.cpp` file in the `src/` folder is the entry point for the simulation. It parses the command line arguments and
then runs the simulation.

## <a name="tests"></a> 4. Tests
After running `make all` you can run `make test` and this will run the entire GoogleTest testing suite within the `tests/` folder.

Ensure that your local machine can handle generating up to 6 MPI ranks, as the tests in `tests/test_mpi_comms.cpp`
attempts to run a small simulation across 6 ranks.

## <a name="docs"></a> 5. Documentation
The documentation is generated by doxygen, run:
```bash
make docs
```
And then open `docs/html/index.html` in your browser.


## <a name="report"></a> 6. Report
The report is located at [report/out/main.pdf](report/out/main.pdf).

## <a name="auto"></a> 7. Use of auto-generation tools
Auto-generation tools, specifically ChatGPT and co-pilot were used in a few instances throughout the project as
detailed here:
 - Co-pilot was used to assist in writing a lot of the docstrings via tab completion. These were amended as required.
 - ChatGPT was used heavily in creating the shell scripts in the `scripts/` folder. The bash function `run_simulation`
was almost entirely written by ChatGPT, with a few minor edits.
 - ChatGPT was used to assist in a few matplotlib plot formatting queries.

