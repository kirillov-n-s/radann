#pragma once

namespace radann::cuda
{
    class prop
    {
    private:
        cudaDeviceProp* _handle;
        prop();

    public:
        prop(const prop&) = delete;
        ~prop();
        friend cudaDeviceProp* get_prop();
    };
}

namespace radann::cuda
{
    prop::prop()
        : _handle(new cudaDeviceProp())
    {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(_handle, device);
    }

    prop::~prop()
    {
        delete _handle;
    }

    cudaDeviceProp* get_prop()
    {
        static prop context;
        return context._handle;
    }
}
