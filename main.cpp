#include "matrix.hpp"
#include "nn.hpp"
#include "utility.hpp"
#include <cstdlib>
#include <iostream>

int main() {
    float lr = 1e-3;  

    Matrix x(2,1);
    Dense dlayer(2, 2);
    Dense dlayer2(2, 1);
    ReLU re;
    for(int i = 0; i < 1000000; i++) {
        int r = i % 4;

        Matrix target(1,1);
        if(r == 0) { x[0][0] = 0; x[1][0] = 0; target[0][0] = 0; }
        if(r == 1) { x[0][0] = 0; x[1][0] = 1; target[0][0] = 1; }
        if(r == 2) { x[0][0] = 1; x[1][0] = 0; target[0][0] = 1; }
        if(r == 3) { x[0][0] = 1; x[1][0] = 1; target[0][0] = 0; }        
        
        // Matrix of predictions
        Matrix pred = dlayer.forward(x);
        Matrix pred_act = re.forward(pred);
        Matrix out = dlayer2.forward(pred_act);

        // Loss gradient
        Matrix delta(out.rows, out.cols);
        for(size_t i = 0; i < delta.rows; i++) {
            for(size_t j = 0; j < delta.cols; j++) {
                delta[i][j] = 2 * (out[i][j] - target[i][j]);
            }
        }

        // Backpropagating 
        Matrix grad_act = dlayer2.backward(delta, lr);
        Matrix grad = re.backward(grad_act);
        dlayer.backward(grad, lr);
    }
   
    Matrix y = dlayer2.forward(re.forward(dlayer.forward(x)));
    // std::cout << "size(row x col): " << y.rows << " x " << y.cols << "\n";
    std::cout << x[0][0] << "^" << x[1][0] << " = " << y[0][0] << std::endl;
    Matrix n(2, 1);
    n[0][0] = 1;
    n[1][0] = 0;
    Matrix z = dlayer2.forward(re.forward(dlayer.forward(n)));
    std::cout << n[0][0] << "^" << n[1][0] << " = " << z[0][0] << std::endl;
}
