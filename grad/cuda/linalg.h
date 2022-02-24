#pragma once
#include "context.h"
#include "../meta/meta.h"

namespace grad::cuda
{
    template<typename T>
    struct linalg
    {
        static void dot(const T*, const T*, T*, size_t);
        static void norm2(const T*, T*, size_t);

        static void gemv(const T*, const T*, T*, size_t, size_t);
        static void ger(const T*, const T*, T*, size_t, size_t);

        static void gemm(const T*, const T*, T*, size_t, size_t, size_t);
        static void trans(const T*, T*, size_t, size_t);
        //void inv();
    };
}

namespace grad::cuda
{
    template<typename T>
    void linalg<T>::dot(const T *lhs, const T *rhs, T *res, size_t size)
    {
        auto& handle = global_context().cublas;
        cublasStatus_t status;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSdot(handle,
                                size,
                                lhs, 1,
                                rhs, 1,
                                res);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDdot(handle,
                                size,
                                lhs, 1,
                                rhs, 1,
                                res);
        else
            static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS dot failed. cuBLAS error status " + std::to_string(status));
    }

    template<typename T>
    void linalg<T>::norm2(const T *arg, T *res, size_t size)
    {
        auto& handle = global_context().cublas;
        cublasStatus_t status;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSnrm2(handle,
                                 size,
                                 arg, 1,
                                 res);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDnrm2(handle,
                                 size,
                                 arg, 1,
                                 res);
        else
            static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS nrm2 failed. cuBLAS error status " + std::to_string(status));
    }

    template<typename T>
    void linalg<T>::gemv(const T *lhs, const T *rhs, T *res, size_t rows, size_t cols)
    {
        auto& handle = global_context().cublas;
        cublasStatus_t status;
        const T alpha = 1.;
        const T beta = 0.;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSgemv(handle,
                                 CUBLAS_OP_N,
                                 rows, cols,
                                 &alpha,
                                 lhs, rows,
                                 rhs, 1,
                                 &beta,
                                 res, 1);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDgemv(handle,
                                 CUBLAS_OP_N,
                                 rows, cols,
                                 &alpha,
                                 lhs, rows,
                                 rhs, 1,
                                 &beta,
                                 res, 1);
        else
            static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS gemv failed. cuBLAS error status " + std::to_string(status));
    }

    template<typename T>
    void linalg<T>::ger(const T *lhs, const T *rhs, T *res, size_t rows, size_t cols)
    {
        auto& handle = global_context().cublas;
        cublasStatus_t status;
        const T alpha = 1.;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSger(handle,
                                rows, cols,
                                &alpha,
                                lhs, 1,
                                rhs, 1,
                                res, rows);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDger(handle,
                                rows, cols,
                                &alpha,
                                lhs, 1,
                                rhs, 1,
                                res, rows);
        else
            static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS ger failed. cuBLAS error status " + std::to_string(status));
    }

    template<typename T>
    void linalg<T>::gemm(const T *lhs, const T *rhs, T *res, size_t rows, size_t mid, size_t cols)
    {
        auto& handle = global_context().cublas;
        cublasStatus_t status;
        const T alpha = 1.;
        const T beta = 0.;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 rows, cols, mid,
                                 &alpha,
                                 lhs, rows,
                                 rhs, mid,
                                 &beta,
                                 res, rows);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 rows, cols, mid,
                                 &alpha,
                                 lhs, rows,
                                 rhs, mid,
                                 &beta,
                                 res, rows);
        else
            static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS gemm failed. cuBLAS error status " + std::to_string(status));
    }

    template<typename T>
    void linalg<T>::trans(const T *arg, T *res, size_t rows, size_t cols)
    {
        auto& handle = global_context().cublas;
        cublasStatus_t status;
        const T alpha = 1.;
        const T beta = 0.;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSgeam(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 cols, rows,
                                 &alpha,
                                 arg, rows,
                                 &beta,
                                 nullptr, cols,
                                 res, cols);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDgeam(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,
                                 cols, rows,
                                 &alpha,
                                 arg, rows,
                                 &beta,
                                 nullptr, cols,
                                 res, cols);
        else
            static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS geam failed. cuBLAS error status " + std::to_string(status));
    }
}
