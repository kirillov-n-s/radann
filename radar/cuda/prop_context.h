#pragma once

namespace radar::cuda
{
    class prop_context
    {
    private:
        cudaDeviceProp* _handle;
        prop_context();

    public:
        prop_context(const prop_context&) = delete;
        ~prop_context();
        friend cudaDeviceProp* get_prop();
    };
}

namespace radar::cuda
{
    prop_context::prop_context()
        : _handle(new cudaDeviceProp())
    {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(_handle, device);
    }

    prop_context::~prop_context()
    {
        delete _handle;
    }

    cudaDeviceProp* get_prop()
    {
        static prop_context context;
        return context._handle;
    }
}
