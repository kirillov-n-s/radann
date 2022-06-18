# Radann

Radann is _(hopefully going to be, some day)_ a modern C++ library for multidimentional array handling and automatic differentiation,
implemented using smart expression templates with GPU parallelism and calls to optimized CUDA libraries.

As for now, it is a university coursework with a lot of unfinished and "just working" stuff that is subject to change.

The library is contained in `radann` folder and only needs `radann.h` to be included to use.
Requires C++17 and CUDA Toolkit (developed on 11.6).
The repository also contains a neural network example for MNIST digit recognition in `main.cu`.
