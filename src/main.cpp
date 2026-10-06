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

    model.add_layer<Dense>(1, 16);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(16, 1);
    size_t batch_size = 1000;
    Matrix x(1,batch_size);
    Matrix target(1,batch_size);
    for(size_t k = 0; k < batch_size; k++) {
        float v = randf() * 2.0f - 1.0f;
        x[0][k] = v;
        target[0][k] = v * v;
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
    Matrix test(1, 1);
    test[0][0] = 0.5;
    Matrix out = model.forward(test);
    std::cout << "0.5 ^ 2 = " << out[0][0] << std::endl;
}
