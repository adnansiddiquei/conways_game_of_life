# Makefile adapted from J. R. Fergusson's Makefile from Course GitLab
BUILDS = release debug build clean

.PHONY := $(BUILDS)

.DEFAULT_GOAL := release

release :
	@echo "Compiling Release"
	cmake --build build/release
	@echo "Done Compiling Release"

build :
	@echo "Building CMake Projects"
	cmake -S . -B build/release -DCMAKE_BUILD_TYPE=Release
	@echo "Done Building CMake Projects"

clean :
	@echo "Cleaning CMake Projects"
	rm -rf build
	@echo "Done Cleaning CMake Projects"
	@echo "You need to run 'make build' to build the Projects again."

all : build release
	cp ./build/release/src/main main