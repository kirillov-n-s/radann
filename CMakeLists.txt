cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu
        grad/shape.h
        grad/array.h
        grad/cuda/storage.h
        grad/engine/expr.h
        grad/functor/unary.h
        grad/unary.h
        grad/functor/binary.h
        grad/binary.h
        grad/functor/ext.h
        grad/cuda/kernel/assign.h
        grad/cuda/assign.h
        grad/engine/unary.h
        grad/engine/binary.h
        grad/engine/access.h
        grad/engine/term.h
        grad/functor/sequence.h
        grad/sequence.h
        grad/functor/random.h
        grad/random.h
        grad/cuda/random.h
        grad/grad.h
        grad/meta/meta.h
        grad/linalg.h
        grad/cuda/cublas.h
        grad/cuda/cusolver.h
        grad/cuda/launch.h
        grad/creation.h grad/eager_eval.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(coursework cuBLAS cuSOLVER)
