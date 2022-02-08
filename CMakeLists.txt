cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu grad/shape.h grad/array.h grad/storage.h grad/ops/expr.h grad/ops/unary_functors.h grad/unary_ops.h grad/ops/binary_functors.h grad/binary_ops.h grad/ops/unary_functors_ext.h grad/kernels/assign.h grad/assign.h grad/ops/unary.h grad/ops/binary.h grad/ops/access.h grad/constant.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
