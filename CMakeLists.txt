cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu grad/shape.h grad/array.h grad/storage.h grad/kernel.h grad/xtmp/expr.h grad/xtmp/unary_ftors_base.h grad/unary_ops.h grad/xtmp/binary_ftors_base.h grad/binary_ops.h grad/xtmp/unary_ftors_ext.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
