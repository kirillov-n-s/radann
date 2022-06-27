# Radann

Radann is _(hopefully going to be, some day)_ a modern C++ library for multidimentional array handling and automatic differentiation,
implemented using smart expression templates with GPU parallelism and calls to optimized CUDA libraries.

As for now, it is a university coursework with a lot of unfinished and "just working" stuff that is subject to change.
Possible improvements:
- Engine
  * variadic argument tuples instead of unary & binary
  * memory pooling (minimize alocations)
  * plans and delayed execution (minimize kernel launches)
- Autodiff
  * tape context management (start, stop, pause, multiple tapes)
  * differentiated batch operations (like: matrix + column-vector)
  * differentiable/non-differentiable marker on operations
- Array
  * comparison, boolean specialization and operations
  * conditional access, dimension skip and slicing
  * concatenation
  * format choice (row-/column-major)
- Operations
  * cuBLAS transpose (possibly lazy w.r.t. matmul)
  * reduce by dimension
  * custom reduce kernel instead of `thrust::reduce_by_key` (shared memory block reduce)
  * cuRAND state caching
  * cuDNN convolution, attention, etc.
  * cuSOLVER matrix inverse

The library is contained in `radann` folder and only needs `radann.h` to be included to use.
Requires C++17 and CUDA Toolkit (developed on 11.6).
The repository also contains a neural network example for MNIST digit recognition in `main.cu`.
