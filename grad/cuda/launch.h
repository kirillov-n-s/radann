#pragma once

namespace grad::cuda
{
    template <typename Kernel>
    void get_launch_config(Kernel, size_t, size_t, size_t&, size_t&);

    template <typename Kernel, typename Func, std::enable_if_t<!std::is_integral_v<Func>, void*> = nullptr>
    void get_launch_config(Kernel, size_t, Func, size_t&, size_t&);
}

namespace grad::cuda
{
    template <typename Kernel>
    void get_launch_config(Kernel kernel,
                           size_t problem_size, size_t n_shared_bytes,
                           size_t& block_dim, size_t& grid_dim)
    {
        int adviced_block_dim, min_grid_dim;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_dim,
                &adviced_block_dim,
                kernel,
                n_shared_bytes,
                problem_size);
        block_dim = adviced_block_dim;
        grid_dim = (problem_size - 1) / block_dim + 1;
    }

    template <typename Kernel, typename Func, std::enable_if_t<!std::is_integral_v<Func>, void*>>
    void get_launch_config(Kernel kernel,
                           size_t problem_size, Func block_dim_to_n_shared_bytes,
                           size_t& block_dim, size_t& grid_dim)
    {
        int adviced_block_dim, min_grid_dim;
        cudaOccupancyMaxPotentialBlockSizeVariableSMem(
                &min_grid_dim,
                &adviced_block_dim,
                kernel,
                block_dim_to_n_shared_bytes,
                problem_size);
        block_dim = adviced_block_dim;
        grid_dim = (problem_size - 1) / block_dim + 1;
    }
}
