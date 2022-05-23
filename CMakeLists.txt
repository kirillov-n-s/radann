cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu
        radann/core/shape.h
        radann/core/array.h
        radann/cuda/shared_storage.h
        radann/engine/expr.h
        radann/functor/unary.h
        radann/core/unary.h
        radann/functor/binary.h
        radann/core/binary.h
        radann/cuda/kernel/assign.h
        radann/cuda/assign.h
        radann/engine/unary_lazy.h
        radann/engine/binary_lazy.h
        radann/engine/access.h
        radann/engine/term.h
        radann/functor/sequence.h
        radann/core/sequence.h
        radann/functor/random.h
        radann/core/random.h
        radann/cuda/random.h
        radann/radann.h
        radann/core/linalg.h
        radann/cuda/cublas.h
        radann/cuda/cusolver.h
        radann/core/creation.h
        radann/core/eval.h
        radann/cuda/host_buffer.h
        radann/engine/unary_eager.h
        radann/functor/linalg.h
        radann/cuda/reduce.h
        radann/core/reduce.h
        radann/core/util.h
        radann/functor/reduce.h
        radann/engine/binary_eager.h
        radann/core/default.h
        radann/cuda/prop.h
        radann/engine/tape.h
        radann/engine/tape_context.h
        radann/cuda/shared_array.h
        radann/cuda/unique_array.h
        radann/cuda/launch.h radann/core/tape.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(coursework cuBLAS cuSOLVER)
