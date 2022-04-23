cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu
        grad/core/shape.h
        grad/core/array.h
        grad/cuda/shared_storage.h
        grad/engine/expr.h
        grad/functor/unary.h
        grad/unary.h
        grad/functor/binary.h
        grad/binary.h
        grad/functor/extension.h
        grad/cuda/kernel/assign.h
        grad/cuda/assign.h
        grad/engine/unary_lazy.h
        grad/engine/binary_lazy.h
        grad/engine/access.h
        grad/engine/term.h
        grad/functor/sequence.h
        grad/sequence.h
        grad/functor/random.h
        grad/random.h
        grad/cuda/random.h
        grad/grad.h
        grad/linalg.h
        grad/cuda/cublas.h
        grad/cuda/cusolver.h
        grad/creation.h
        grad/core/eval.h
        grad/cuda/host_buffer.h
        grad/engine/unary_eager.h
        grad/functor/linalg.h
        grad/cuda/reduce.h
        grad/reduce.h
        grad/util/util.h
        grad/functor/reduce.h
        grad/engine/binary_eager.h
        grad/extension.h
        grad/functor/activation.h
        grad/activation.h
        grad/core/defaults.h
        grad/cuda/device_prop.h
        grad/engine/tape.h
        grad/engine/tape_context.h
        grad/cuda/shared_array.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(coursework cuBLAS cuSOLVER)
