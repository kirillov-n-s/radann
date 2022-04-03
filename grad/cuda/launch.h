#pragma once

namespace grad::cuda
{
    template <typename Kernel>
    void get_launch_parameters(Kernel, size_t, int&, int&);
}

namespace grad::cuda
{
    template <typename Kernel>
    void get_launch_parameters(Kernel kernel, size_t problem_size, int& block_dim, int& grid_dim)
    {
        int adviced_block_dim, min_grid_dim;
        cudaOccupancyMaxPotentialBlockSize(
                &min_grid_dim,
                &adviced_block_dim,
                kernel,
                0,
                problem_size);
        block_dim = adviced_block_dim;
        grid_dim = (problem_size - 1) / block_dim + 1;
    }
}
