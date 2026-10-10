#include "include/matrix.hpp"
#include "include/nn.hpp"
#include "include/model.hpp"
#include "include/utility.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <utility>

#define ITERATIONS 20000

int main() {
    float lr = 0.1f;
    Model model;
    MSELoss loss;
    float l;

    model.add_layer<Dense>(2, 32);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(32, 1);
    size_t batch_size = 1000;
    Tensor x({2,batch_size});
    Tensor target({1,batch_size});

    for(size_t k = 0; k < batch_size; k++) {
        float a = randf() * 2.0f - 1.0f;
        float b = randf() * 2.0f - 1.0f;

        x.at(0, k) = a;
        x.at(1, k) = b;
        target.at(0, k) = a * b;
    }

    for(int i = 0; i < ITERATIONS; i++) {
        // Matrix of predictions
        Tensor out = model.forward(x);

        // Loss gradient
        l = loss.forward(out, target);
        Tensor delta = loss.backward(out, target);

       // Backpropagating
        model.backward(delta);

        // Updating weights and biases on all layers
        model.update(lr);
        if(i % 100 == 0) {
            std::cout << "loss = " << l << "\n";
            std::cout << ((float)i / (float)ITERATIONS) * 100.f << "%\n";
            std::cout << "\033[H\033[J";
        }
    }

    // Test the trained model on values covering the complete input range.
    constexpr std::array<std::pair<float, float>, 13> test_cases {{
        { 0.25f,  0.11f},
        {-0.25f,  0.11f},
        { 0.25f, -0.11f},
        {-0.25f, -0.11f},
        { 0.00f,  0.75f},
        { 0.75f,  0.00f},
        { 0.50f,  0.50f},
        { 0.10f,  0.90f},
        { 0.90f, -0.80f},
        { 1.00f,  1.00f},
        {-1.00f,  1.00f},
        { 1.00f, -1.00f},
        {-1.00f, -1.00f},
    }};

    Tensor test({2, test_cases.size()});
    for(std::size_t i = 0; i < test_cases.size(); i++) {
        test.at(0, i) = test_cases[i].first;
        test.at(1, i) = test_cases[i].second;
    }

    Tensor predictions = model.forward(test);
    float squared_error_sum = 0.0f;
    float absolute_error_sum = 0.0f;
    float maximum_absolute_error = 0.0f;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nValidation results:\n";
    for(std::size_t i = 0; i < test_cases.size(); i++) {
        const auto [a, b] = test_cases[i];
        const float expected = a * b;
        const float predicted = predictions.at(0, i);
        const float error = predicted - expected;
        const float absolute_error = std::abs(error);

        squared_error_sum += error * error;
        absolute_error_sum += absolute_error;
        maximum_absolute_error =
            std::max(maximum_absolute_error, absolute_error);

        std::cout << a << " * " << b
                  << " | expected: " << expected
                  << " | predicted: " << predicted
                  << " | abs error: " << absolute_error << '\n';
    }

    const float test_count = static_cast<float>(test_cases.size());
    std::cout << "\nValidation MSE: "
              << squared_error_sum / test_count << '\n';
    std::cout << "Validation MAE: "
              << absolute_error_sum / test_count << '\n';
    std::cout << "Maximum absolute error: "
              << maximum_absolute_error << '\n';
}
