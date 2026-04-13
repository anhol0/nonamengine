#include "matrix.hpp"
#include "nn.hpp"
#include "model.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    float lr = 1e-3;  

    Matrix x(2,1);
    Model model;
    model.add_layer<Dense>(2, 4);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(4, 1);
    for(int i = 0; i < 100000; i++) {
        int r = rand() % 4;

        Matrix target(1,1);
        if(r == 0) { x[0][0] = 0; x[1][0] = 0; target[0][0] = 0; }
        if(r == 1) { x[0][0] = 0; x[1][0] = 1; target[0][0] = 1; }
        if(r == 2) { x[0][0] = 1; x[1][0] = 0; target[0][0] = 1; }
        if(r == 3) { x[0][0] = 1; x[1][0] = 1; target[0][0] = 0; }        
        
        // Matrix of predictions
        Matrix out = model.forward(x);

        // Loss gradient
        Matrix delta = model.gradient(out, target);

        // Backpropagating  
        model.backward(delta);

        // Updating weights and biases on all layers
        model.update(lr);
    }

    // Testing
    for(int a = 0; a < 2; a++) {
        for(int b = 0; b < 2; b++) {
            Matrix t(2,1);
            t[0][0] = a;
            t[1][0] = b;

            Matrix o = model.forward(t); 
            std::cout << a << "^" << b << " = " << o[0][0] << "\n";
        }
    }
}
