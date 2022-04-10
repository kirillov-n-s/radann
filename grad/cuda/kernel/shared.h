#pragma once

namespace grad::cuda::kernel
{
    template <typename T>
    __device__ T* extern_shared_memory();
}

namespace grad::cuda::kernel
{
    template <typename T>
    __device__ T* extern_shared_memory()
    {
        extern __shared__ unsigned char memory[];
        return reinterpret_cast<T*>(memory);
    }
}
