#include "matrix.hpp"
#include "nn.hpp"
#include "model.hpp"
#include <cmath>
#include <iostream>

#define ITERATIONS 2000000

int main() {
    float lr = 1e-3;  
    Matrix x(2,1);
    Model model;
    MSELoss loss;
    float l;
    model.add_layer<Dense>(2, 64);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(64, 1);
    for(int i = 0; i < ITERATIONS; i++) {
        // Target data
        Matrix target(1,1);

        // Training data
        x[0][0] = (rand() % 20)/ 19.f;
        x[1][0] = (rand() % 20) / 19.f;
        target[0][0] = (x[0][0] * x[1][0]);

        // Matrix of predictions
        Matrix out = model.forward(x);

        // Loss gradient
        l = loss.forward(out, target);
        Matrix delta = loss.backward(out, target);

       // Backpropagating  
        model.backward(delta);

        // Updating weights and biases on all layers
        model.update(lr);
        if(i % 10000 == 0) {
            std::cout << "loss = " << l << "\n";
            std::cout << ((float)i / (float)ITERATIONS) * 100.f << "%\n";
            std::cout << "\033[H\033[J";
        } 
    }   

    // Testing
    for(int a = 0; a < 10; a++) {
        for(int b = 0; b < 10; b++) {
            Matrix t(2,1);
            t[0][0] = a / 19.f;
            t[1][0] = b / 19.f;

            Matrix o = model.forward(t); 
            std::cout << a << "*" << b << " = " << o[0][0] * 19.f*19.f << "\n";
        }
    }

    Matrix t(2,1);
    t[0][0] = 11.f / 19.f;
    t[1][0] = 11.f / 19.f;

    Matrix o = model.forward(t); 
    std::cout << 11 << "*" << 11 << " = " << o[0][0] * 19.f*19.f << "\n";
}
