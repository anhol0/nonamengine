#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include "utility.hpp"

class Dense {
public:
    Matrix weights, biases;
    Matrix input_cache;

    Dense(size_t in_features, size_t out_features): 
        weights(out_features, in_features), 
        biases(out_features, 1), 
        input_cache(in_features, 1) 
    {
        for(size_t i = 0; i < weights.rows; i++) {
            for(size_t j = 0; j < weights.cols; j++) {
                weights[i][j] = (randf() * 2.0f - 1.0f)* 0.1f;
            }
        } 
        for(size_t i = 0; i < biases.rows; i++) {
            for(size_t j = 0; j < biases.cols; j++) {
                biases[i][j] = 0.0f;
            }
        }
    }
    Matrix forward(const Matrix &input) {
        input_cache = input;
        return weights * input + biases; 
    }
    Matrix backward(const Matrix &grad_output, float lr) {
        Matrix inputs_T = transpose(input_cache);
        Matrix grad_weights = grad_output * inputs_T;

        Matrix grad_biases(biases.rows, biases.cols);
        for(size_t i = 0; i < grad_output.rows; i++) {
            float sum = 0.0f;
            for(size_t j = 0; j < grad_output.cols; j++) {
                sum += grad_output[i][j];
            }
            grad_biases[i][0] = sum;
        }
        Matrix grad_inputs = transpose(weights) * grad_output; 
        weights = weights - (grad_weights * lr);
        biases = biases - (grad_biases * lr);

        return grad_inputs;

    }
};

class ReLU {
public:
    Matrix cache;
    Matrix forward(const Matrix &input) {
        cache = input;
        return relu(input);
    }
    Matrix backward(const Matrix &grad_output) {
        Matrix grad(cache.rows, cache.cols);
        for(size_t i = 0; i < cache.rows; i++) {
            for(size_t j = 0; j < cache.cols; j++) {
                grad[i][j] = cache[i][j] > 0 ? grad_output[i][j] : 0.0f;
            }
        }
        return grad;
    }
};

#endif
