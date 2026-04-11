#include "matrix.hpp"
#include "nn.hpp"
#include <cstdlib>
#include <iostream>

void print_matrix(const Matrix &m) {
    for(int i = 0; i < m.rows; i++) {
        for(int j = 0; j < m.cols; j++) {
            std::cout << m[i][j] << ", ";
        }
        std::cout << "\n";
    }
}

float randf() {
    return (float)rand() / RAND_MAX;
}

int main() {
    float lr = 1e-3; 
    Matrix weights(1, 3);
    Matrix biases(1, 1);
    for(int i = 0; i < weights.rows; i++) {
        for(int j = 0; j < weights.cols; j++) {
            weights[i][j] = (randf() * 2.0f - 1.0f) * 0.1f;
        }
    }
    biases[0][0] = 0.0f;
    Matrix x(3, 1);
    for(int i = 0; i < 300000; i++) {
        x[0][0] = rand()%10;
        x[1][0] = rand()%10;
        x[2][0] = rand()%10;

        Matrix target(1, 1);
        target[0][0] = x[0][0] + x[1][0]+x[2][0];
        Matrix pred = layer(x, weights, biases);
        float l = loss(pred, target);
        BackpropResult grads = backprop(x, weights, target, pred, biases);
        weights = weights - (grads.grad_weights * lr);
        biases = biases - (grads.grad_biases * lr);

        if(i % 100 == 0) {
            std::cout << "loss: " << l << "\n";
        }
    }
    Matrix y = layer(x, weights, biases);
    std::cout << x[0][0] << " + " << x[1][0] << " + " << x[2][0] << " = " << y[0][0] << std::endl;
    Matrix n(3, 1);
    n[0][0] = 2;
    n[1][0] = 60;
    n[2][0] = 15;
    Matrix z = layer(n, weights, biases);
    std::cout << n[0][0] << " + " << n[1][0] << " + " << n[2][0] << " = " << z[0][0] << std::endl;
}
