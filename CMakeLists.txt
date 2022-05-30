cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu
        radann/core/shape.h
        radann/core/array.h
        radann/cuda/shared_storage.h
        radann/expr/base.h
        radann/func/unary.h
        radann/core/unary.h
        radann/func/binary.h
        radann/core/binary.h
        radann/cuda/kernel/assign.h
        radann/cuda/assign.h
        radann/expr/unary.h
        radann/expr/binary.h
        radann/expr/access.h
        radann/expr/term.h
        radann/func/sequence.h
        radann/core/sequence.h
        radann/func/random.h
        radann/core/random.h
        radann/cuda/random.h
        radann/radann.h
        radann/core/linalg.h
        radann/cuda/cublas.h
        radann/cuda/cusolver.h
        radann/core/creation.h
        radann/core/eval.h
        radann/cuda/host_buffer.h
        radann/func/linalg.h
        radann/cuda/reduce.h
        radann/core/reduce.h
        radann/core/util.h
        radann/func/reduce.h
        radann/core/default.h
        radann/cuda/prop.h
        radann/expr/tape.h
        radann/expr/tape_context.h
        radann/cuda/shared_array.h
        radann/cuda/unique_array.h
        radann/cuda/launch.h
        radann/core/tape.h
        radann/expr/eager.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(coursework cuBLAS cuSOLVER)
