#pragma once
#include <string>
#include <cublas_v2.h>
#include "../meta/meta.h"

namespace radann::cuda
{
    class cublas
    {
    private:
        cublasHandle_t _handle;
        cublas();

    public:
        cublas(const cublas&) = delete;
        ~cublas();
        friend cublasHandle_t get_cublas();

        template<typename T>
        static void dot(const T*, const T*, T*, size_t);
        template<typename T>
        static void nrm2(const T*, T*, size_t);

        template<bool Trans, typename T>
        static void gemv(const T*, const T*, T*, size_t, size_t);
        template<typename T>
        static void ger(const T*, const T*, T*, size_t, size_t);

        template<bool LTrans, bool RTrans, typename T>
        static void gemm(const T*, const T*, T*, size_t, size_t, size_t);
        template<typename T>
        static void geam(const T*, T*, size_t, size_t);
    };
}

namespace radann::cuda
{
    cublas::cublas()
    {
        auto status = cublasCreate(&_handle);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS create failed. cuBLAS error status " + std::to_string(status));
    }

    cublas::~cublas()
    {
        cublasDestroy(_handle);
    }

    cublasHandle_t get_cublas()
    {
        static cublas context;
        return context._handle;
    }

    template<typename T>
    void cublas::dot(const T *lhs, const T *rhs, T *res, size_t size)
    {
        auto handle = get_cublas();
        cublasStatus_t status;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

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
    void cublas::nrm2(const T *arg, T *res, size_t size)
    {
        auto handle = get_cublas();
        cublasStatus_t status;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

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

    template<bool Trans, typename T>
    void cublas::gemv(const T *lhs, const T *rhs, T *res, size_t rows, size_t cols)
    {
        auto handle = get_cublas();
        cublasStatus_t status;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        const T alpha = 1.;
        const T beta = 0.;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSgemv(handle,
                                 Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 rows, cols,
                                 &alpha,
                                 lhs, rows,
                                 rhs, 1,
                                 &beta,
                                 res, 1);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDgemv(handle,
                                 Trans ? CUBLAS_OP_T : CUBLAS_OP_N,
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
    void cublas::ger(const T *lhs, const T *rhs, T *res, size_t rows, size_t cols)
    {
        auto handle = get_cublas();
        cublasStatus_t status;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
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

    template<bool LTrans, bool RTrans, typename T>
    void cublas::gemm(const T *lhs, const T *rhs, T *res, size_t rows, size_t mid, size_t cols)
    {
        auto handle = get_cublas();
        cublasStatus_t status;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
        const T alpha = 1.;
        const T beta = 0.;

        if constexpr(std::is_same_v<T, float>)
            status = cublasSgemm(handle,
                                 LTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 RTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 rows, cols, mid,
                                 &alpha,
                                 lhs, LTrans ? mid : rows,
                                 rhs, RTrans ? cols : mid,
                                 &beta,
                                 res, rows);
        else if constexpr(std::is_same_v<T, double>)
            status = cublasDgemm(handle,
                                 LTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 RTrans ? CUBLAS_OP_T : CUBLAS_OP_N,
                                 rows, cols, mid,
                                 &alpha,
                                 lhs, LTrans ? mid : rows,
                                 rhs, RTrans ? cols : mid,
                                 &beta,
                                 res, rows);
        else
                static_assert(meta::always_false_v<T>, "cuBLAS not specialized for this type.");

        cudaDeviceSynchronize();
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS gemm failed. cuBLAS error status " + std::to_string(status));
    }

    template<typename T>
    void cublas::geam(const T *arg, T *res, size_t rows, size_t cols)
    {
        auto handle = get_cublas();
        cublasStatus_t status;
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
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
