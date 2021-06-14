# Solving EIT via PDE constrained optimization

This is a python implementation of a optimization-based estimation of the EIT problem, using PDE constrained optimization, without regularization. 

## Requirements: 

-numpy (with MKL backend)

-scipy 

-numba 

-pypardiso

For some of the examples the scipy.io library to extract the data (already contained in the tests folder)

## Installation

To install just go to the main directory and install as usual: 

python setup.py install

This will install the package on your local python environment (we strongly recommend to create a virtual environment before hand)

## Tests

We have added a context.py script in the test folder, this would allow to run the test files without having to install the package on your local environment. 
