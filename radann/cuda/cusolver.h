#pragma once
#include <cusolverDn.h>

namespace radann::cuda
{
    class cusolver
    {
    private:
        cusolverDnHandle_t _handle;
        cusolver();

    public:
        cusolver(const cusolver&) = delete;
        ~cusolver();
        friend cusolverDnHandle_t get_cusolver();

        /*template<typename trans>
        static void solve(const trans*, const trans*, trans*, size_t);*/
    };
}

namespace radann::cuda
{
    cusolver::cusolver()
    {
        auto status = cusolverDnCreate(&_handle);
        if (status != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("cuSOLVER create failed. cuSOLVER error status " + std::to_string(status));
    }

    cusolver::~cusolver()
    {
        cusolverDnDestroy(_handle);
    }

    cusolverDnHandle_t get_cusolver()
    {
        static cusolver context;
        return context._handle;
    }

    /*template<typename trans>
    void cusolver::solve(const trans *arg, trans *result, size_t size)
    {

    }*/
}