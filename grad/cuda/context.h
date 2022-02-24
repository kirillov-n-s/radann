#pragma once
#include "cublas_v2.h"

namespace grad::cuda
{
    struct context
    {
        cublasHandle_t cublas;

        context();
        ~context();
    };

    context& global_context();
}

namespace grad::cuda
{
    context::context()
    {
        auto status = cublasCreate(&cublas);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS create failed. cuBLAS error status " + std::to_string(status));
    }

    context::~context()
    {
        auto status = cublasDestroy(cublas);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cuBLAS destroy failed. cuBLAS error status " + std::to_string(status));
    }

    context& global_context()
    {
        static context context;
        return context;
    }
}
