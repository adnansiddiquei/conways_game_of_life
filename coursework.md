# Assessed Coursework

## Guidance
The coursework will be submitted via a GitLab repository which we will create for you.  You should place all your code and your report in this repository. The report should be in PDF format in a folder called "report".  You will be provided access to the repository until 11:59pm on Thursday the 28th of March.  After this you will lose access which will constitute submission of your work.

## The problem
Please write a programme that simulates [Conway's game of life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life).

The programme should be able to take as input an integer binary square grid, or generate one with random allocation, then evolve it for a number of time steps given at runtime.  Boundaries are periodic.  The code must be able to handle very large grids buy utilising MPI and domain decomposition to break the problem up.

The calculation on each MPI rank should be optimised and parallelised to make best use of the computing resources available.  You may assume that the memory constraints of the project have been solved by domain decomposition so you are free to make optimisations that trade speed for increased memory use.

The code should be simple to use, written in C++, and come with appropriate example submission scripts for CSD3.

## Criteria
The goal of this coursework project is for you to demonstrate that you understand parallelisation and optimisation practices for High Performance Computing.  

The key skills you will be expected to demonstrate are:
- Flexible and robust MPI domain decomposition of the grid and communication of the boundaries at each step.
- Effective Thread parallelisation using OpenMP
- Optimised routines for the convolution required to count neighbours, it may be helpful to realise that this kernel is separable (Google: separable 2D convolution if this is confusing)
- Rigorous performance testing with appropriate plots and tables showing the results

While the focus of this project is performance we still expect you to demonstrate the following software development best practice:
- Writing clear readable code that is compliant with a common style guide and uses suitable build management tools. 
- Providing appropriate documentation that is compatible with auto-documentation tools like Doxygen.
- The project must be well structured (sensible folder structure, readme, licence etc..) following standard best practice.
- Appropriate and robust unit tests for automatic validation of your code.
- Uses appropriate version control best practice, including branching for development and testing, and commit hooks to protect "main”.  Given the parallel nature of the project, containerisation is not required.

As this project is designed to test your optimisation skills, **no linear algebra or image processing libraries should be used**.  The expectation is that you will only need: standard libraries, MPI, and OpenMP

## Submission
This repository will contain the working code for the project. This should be accompanied by a short report of not more than 3,000 words describing the project and its development. You should ensure your report is logically structured and touches on the points mentioned above in the assessment criteria. Specifically, I would expect reports to cover the following topics, with the third being the largest section:

- Short Introduction
- Selection of Solution Algorithm and Prototyping
- Development, Experimentation, Profiling and Optimisation
- Summary

The report should be written in latex, with the generated PDF of the report placed in a folder called "report" in the repository.
