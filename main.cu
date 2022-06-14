#include "radann/radann.h"
#include <chrono>
#include <fstream>

using timer = std::chrono::system_clock;

int reverse_endian(int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8)  & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}
radann::array<> read_mnist_images(const char *path)
{
    std::ifstream file { path, std::ios::binary };
    if (!file.is_open())
        throw std::runtime_error("Cannot open file " + std::string(path));

    int magic = 0,
        n_images = 0,
        n_rows = 0,
        n_cols = 0;

    file.read((char*)&magic, sizeof(magic));
    //magic = reverse_endian(magic);
    file.read((char*)&n_images, sizeof(n_images));
    n_images = reverse_endian(n_images);
    file.read((char*)&n_rows, sizeof(n_rows));
    n_rows = reverse_endian(n_rows);
    file.read((char*)&n_cols, sizeof(n_cols));
    n_cols = reverse_endian(n_cols);

    auto n = n_images * n_rows * n_cols;
    std::vector<radann::real> data(n);
    for(int i = 0; i < n; i++)
    {
        unsigned char tmp = 0;
        file.read((char*)&tmp, sizeof(tmp));
        data[i] = (radann::real)tmp;
    }

    return radann::make_array(radann::make_shape(n_rows, n_cols, n_images),
                              data.begin(), data.end(),
                              false);
}

radann::array<> read_mnist_labels(const char *path)
{
    std::ifstream file { path, std::ios::binary };
    if (!file.is_open())
        throw std::runtime_error("Cannot open file " + std::string(path));

    int magic = 0,
        n_labels = 0,
        n_digits = 10;

    file.read((char*)&magic, sizeof(magic));
    //magic = reverse_endian(magic);
    file.read((char*)&n_labels, sizeof(n_labels));
    n_labels = reverse_endian(n_labels);

    std::vector<radann::real> data(n_labels * n_digits);
    for(int i = 0; i < n_labels; i++)
    {
        unsigned char tmp = 0;
        file.read((char*)&tmp, sizeof(tmp));
        for (int j = 0; j < n_digits; j++)
            data[i * n_digits + j] = (radann::real)(j == tmp);
    }

    return radann::make_array(radann::make_shape(n_digits, n_labels),
                              data.begin(), data.end(),
                              false);
}

auto to_digit(const radann::array<> &output)
{
    auto host = output.host();
    return std::max_element(host.begin(), host.end()) - host.begin();
}

class neural_network
{
private:
    std::vector<radann::array<>> _weights;
    std::vector<radann::array<>> _biases;

public:
    neural_network(const std::initializer_list<size_t> &layers)
    {
        auto init = radann::uniform<radann::real>() * 0.6_fC - 0.3_fC;
        auto n = layers.size();
        auto data = layers.begin();
        for (int i = 1; i < n; i++)
        {
            _weights.push_back(radann::make_array(radann::make_shape(data[i], data[i - 1]), init, true));
            _biases.push_back(radann::make_array(radann::make_shape(data[i]), init, true));
        }
    }

    radann::array<> predict(const radann::array<> &input, bool train = false) const
    {
        auto res = input;
        auto n = _weights.size();
        for (int i = 0; i < n; i++)
            res >>= radann::make_array(
                    radann::sigmoid(radann::matmul(_weights[i], res) + _biases[i]),
                    train);
        return res;
    }

    int train(const radann::array<>& inputs, const radann::array<>& labels,
              radann::real learning_rate,
              int min_epochs, int max_epochs, radann::real error_threshold,
              bool verbose = false, int every_nth = 100)
    {
        auto i = 0;
        radann::real error_sum = 0.f,
                     mean_error = 0.f;
        auto lr = radann::constant(learning_rate);
        for (;
             i < max_epochs && (mean_error > error_threshold || i < min_epochs);
             i++)
        {
            const auto& label = labels(i);
            const auto& output = predict(inputs(i), true);
            auto loss = radann::sum(radann::pow2(label - output) / radann::constant<radann::real>(output.size()));
            loss.grad() = 1._fC;

            error_sum += *loss.host().data();
            mean_error = error_sum / i;
            if (verbose && i % every_nth == 0)
                std::cout << "Epoch " << i << ": mean error = " << mean_error << '\n';

            radann::reverse();
            for (auto& weight : _weights)
            {
                weight -= weight.grad() * lr;
                weight.grad().zero();
            }
            for (auto& bias : _biases)
            {
                bias -= bias.grad() * lr;
                bias.grad().zero();
            }
            loss.deactivate_grad();
            radann::clear();
        }

        for (auto& weight : _weights)
            weight.deactivate_grad();
        for (auto& bias : _biases)
            bias.deactivate_grad();
        radann::clear();

        if (verbose)
            std::cout << '\n';

        return i;
    }

    radann::real test(const radann::array<>& inputs, const radann::array<>& labels, int n_tests)
    {
        int n_correct = 0;
        for (int i = 0; i < n_tests; i++)
            n_correct += to_digit(predict(inputs(i))) == to_digit(labels(i));
        return (radann::real)n_correct / n_tests;
    }
};

int main()
{
    std::string dir = R"(C:\Users\user\Desktop\University\!Coursework\coursework\mnist\)";

    auto train_images = read_mnist_images((dir + "train-images.idx3-ubyte").c_str());
    auto train_labels = read_mnist_labels((dir + "train-labels.idx1-ubyte").c_str());
    auto test_images  = read_mnist_images((dir + "t10k-images.idx3-ubyte").c_str());
    auto test_labels  = read_mnist_labels((dir + "t10k-labels.idx1-ubyte").c_str());

    radann::array<> train_images_flattened = train_images.flatten(1) / 255._fC;
    radann::array<> test_images_flattened  = test_images.flatten(1) / 255._fC;

    auto n_inputs  = train_images_flattened.shape(0);
    auto n_outputs = train_labels.shape(0);
    size_t n_hidden1 = 128;
    size_t n_hidden2 = 128;

    auto learning_rate = 4.0f;
    auto min_epochs = 10;
    auto max_epochs = train_images_flattened.shape(1);
    auto error_threshold = 0.03f;
    auto n_tests = test_images_flattened.shape(1);

    neural_network nn { n_inputs, n_hidden1, n_hidden2, n_outputs };

    auto then = timer::now();
    auto train_epochs = nn.train(train_images_flattened, train_labels,
                                 learning_rate,
                                 min_epochs, max_epochs, error_threshold,
                                 true);
    auto train_time = std::chrono::duration_cast<std::chrono::seconds>(timer::now() - then).count();

    then = timer::now();
    auto test_accuracy = nn.test(test_images_flattened, test_labels, n_tests);
    auto test_time = std::chrono::duration_cast<std::chrono::seconds>(timer::now() - then).count();

    std::cout << "Train time = " << train_time << " s\n"
              << "Train epochs = " << train_epochs << '\n'
              << "Test time = " << test_time << " s\n"
              << "Test accuracy = " << test_accuracy << '\n';

    int i = -1;
    do
    {
        std::cout << "\nEnter image index (0..9999): ";
        std::cin >> i;
        if (i == -1)
            break;

        std::cout << test_images(i)
                  << "Label: " << to_digit(test_labels(i)) << '\n'
                  << "Predicted: " << to_digit(nn.predict(test_images_flattened(i))) << "\n\n";
    }
    while (true);

    std::cin.get();
}
