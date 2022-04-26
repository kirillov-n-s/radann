cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu
        radar/core/shape.h
        radar/core/array.h
        radar/cuda/shared_storage.h
        radar/engine/expr.h
        radar/functor/unary.h
        radar/core/unary.h
        radar/functor/binary.h
        radar/core/binary.h
        radar/functor/extension.h
        radar/cuda/kernel/assign.h
        radar/cuda/assign.h
        radar/engine/unary_lazy.h
        radar/engine/binary_lazy.h
        radar/engine/access.h
        radar/engine/term.h
        radar/functor/sequence.h
        radar/core/sequence.h
        radar/functor/random.h
        radar/core/random.h
        radar/cuda/random.h
        radar/radar.h
        radar/core/linalg.h
        radar/cuda/cublas.h
        radar/cuda/cusolver.h
        radar/core/creation.h
        radar/core/eval.h
        radar/cuda/host_buffer.h
        radar/engine/unary_eager.h
        radar/functor/linalg.h
        radar/cuda/reduce.h
        radar/core/reduce.h
        radar/core/util.h
        radar/functor/reduce.h
        radar/engine/binary_eager.h
        radar/core/extension.h
        radar/functor/activation.h
        radar/core/activation.h
        radar/core/default.h
        radar/cuda/prop_context.h
        radar/engine/tape.h
        radar/engine/tape_context.h
        radar/cuda/shared_array.h
        radar/cuda/unique_array.h
        radar/cuda/launch.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(coursework cuBLAS cuSOLVER)
