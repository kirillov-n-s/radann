#include "grad/array.h"

int main()
{
    grad::array<float, 3> x { grad::make_shape(2, 4, 4), { 111, 211, 121, 221, 131, 231, 141, 241,
                                                           112, 212, 122, 222, 132, 232, 142, 242,
                                                           113, 213, 123, 223, 133, 233, 143, 243,
                                                           114, 214, 124, 224, 134, 234, 144, 244, } };
    std::cout << x;
    {
        auto f = x.flatten<1>();
        std::cout << f;
        {
            auto s = x(1);
            std::cout << s;
            s = {-1, -2, -3, -4, -5, -6, -7, -8};
            std::cout << s << x;
            f >>= grad::array<float, 2>{grad::make_shape(4, 4), {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, -1, -2, -3, -4, -5, -6}};
            std::cout << f << x;
            {
                auto y = x.reshape(grad::make_shape(2, 2, 2, 2, 2));
                x >>= f.reshape(grad::make_shape(4, 2, 2));
                s = {0, 0, 0, 0, 0, 0, 0, 0};
                std::cout << x << y << f << s;
            }
            std::cout << s;
        }
        std::cout << f;
    }
    std::cout << x;

    std::cin.get();
}
