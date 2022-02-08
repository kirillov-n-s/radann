#include "grad/array.h"
#include "grad/unary_ops.h"
#include "grad/binary_ops.h"
#include "grad/constant.h"

int main()
{
    grad::array<float, 2> x { grad::make_shape(4, 3), { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 } };
    grad::array<float, 1> y { grad::make_shape(4), { -9, -9, -9, -9 } };
    grad::array<float, 2> z = x + grad::abs(y);

    std::cout << x << y << z;

    std::cin.get();
}
