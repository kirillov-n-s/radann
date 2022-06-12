cmake_minimum_required(VERSION 3.21)
project(coursework CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(coursework main.cu
        radann/core/shape.h
        radann/core/array.h
        radann/cuda/shared_storage.h
        radann/expr/base.h
        radann/oper/unary.h
        radann/func/unary.h
        radann/oper/binary.h
        radann/func/binary.h
        radann/cuda/kernel/assign.h
        radann/cuda/assign.h
        radann/expr/unary.h
        radann/expr/binary.h
        radann/expr/access.h
        radann/expr/term.h
        radann/oper/sequence.h
        radann/func/sequence.h
        radann/oper/random.h
        radann/func/random.h
        radann/cuda/random.h
        radann/radann.h
        radann/func/linalg.h
        radann/cuda/cublas.h
        radann/cuda/cusolver.h
        radann/api/creation.h
        radann/cuda/host_buffer.h
        radann/core/linalg.h
        radann/cuda/reduce.h
        radann/func/reduce.h
        radann/meta/meta.h
        radann/core/reduce.h
        radann/cuda/prop.h
        radann/diff/tape.h
        radann/diff/tape_context.h
        radann/cuda/shared_array.h
        radann/cuda/launch.h
        radann/api/tape.h
        radann/core/eager.h
        radann/diff/grad_unary.h
        radann/diff/grad_binary.h
        radann/diff/propagate.h
        radann/diff/is_ad.h
        radann/diff/strategy_dynamic_ad.h
        radann/diff/backward.h
        radann/diff/strategy_no_ad.h
        radann/api/using.h
        radann/api/eval.h radann/core/default.h radann/diff/entry.h)

set_target_properties(coursework PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(coursework cuBLAS cuSOLVER)
