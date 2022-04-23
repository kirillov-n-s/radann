#pragma once

namespace grad::cuda
{
    class device_prop
    {
    private:
        cudaDeviceProp* _handle;
        device_prop();

    public:
        device_prop(const device_prop&) = delete;
        ~device_prop();
        friend cudaDeviceProp* get_prop();
    };
}

namespace grad::cuda
{
    device_prop::device_prop()
        : _handle(new cudaDeviceProp())
    {
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(_handle, device);
    }

    device_prop::~device_prop()
    {
        delete _handle;
    }

    cudaDeviceProp* get_prop()
    {
        static device_prop context;
        return context._handle;
    }
}
