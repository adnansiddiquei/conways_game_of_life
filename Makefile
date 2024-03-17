# Makefile adapted from J. R. Fergusson's Makefile from Course GitLab
BUILDS = release debug build clean

.PHONY := $(BUILDS)

release :
	@echo "Compiling Release"
	cmake --build build
	@echo "Done Compiling Release"

build :
	@echo "Building CMake Projects"
	cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
	@echo "Done Building CMake Projects"

clean :
	@echo "Cleaning CMake Projects"
	rm -rf build
	rm -rf bin
	@echo "Done Cleaning CMake Projects"
	@echo "You need to run 'make build' to build the Projects again."

all : build release

test :
	mpirun -np 6 ./bin/test_mpi_comms
	ctest --test-dir build --output-on-failure

clean-docs :
	rm -rf docs

docs : clean-docs
	doxygen doxygen.config
