#include "include/matrix.hpp"
#include "include/nn.hpp"
#include "include/model.hpp"
#include "include/utility.hpp"
#include <cmath>
#include <iostream>
#include <ostream>

#define ITERATIONS 30000

int main() {
    float lr = 1e-2;
    Model model;
    MSELoss loss;
    float l;

    model.add_layer<Dense>(2, 32);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(32, 1);
    size_t batch_size = 1000;
    Matrix x(2,batch_size);
    Matrix target(1,batch_size);

    for(size_t k = 0; k < batch_size; k++) {
        float a = randf() * 2.0f - 1.0f;
        float b = randf() * 2.0f - 1.0f;

        x[0][k] = a;
        x[1][k] = b;
        target[0][k] = a * b;
    }

    for(int i = 0; i < ITERATIONS; i++) {
        // Matrix of predictions
        Matrix out = model.forward(x);

        // Loss gradient
        l = loss.forward(out, target);
        Matrix delta = loss.backward(out, target);

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

    // Testing
    Matrix test(2, 1);
    test[0][0] = 0.25;
    test[1][0] = 0.11;
    Matrix out = model.forward(test);
    std::cout << test[0][0] << " * " << test[1][0] << " = " << out[0][0] << std::endl;
}
