#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include "utility.hpp"

class Layer {
public:
    virtual Matrix forward(const Matrix &input) = 0;
    virtual Matrix backward(const Matrix &grad_output) = 0;
    virtual void update(float lr) = 0;
    virtual ~Layer() = default;
};

class Dense : public Layer {
public:
    Matrix weights, biases;
    Matrix grad_weights, grad_biases;
    Matrix input_cache;

    Dense(size_t in_features, size_t out_features);
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &grad_output) override;
    void update(float lr) override;
};

// Class function definitions
inline Dense::Dense(size_t in_features, size_t out_features): 
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

inline Matrix Dense::forward(const Matrix &input) {
    input_cache = input;
    return weights * input + biases; 
}

inline Matrix Dense::backward(const Matrix &grad_output) {
    Matrix inputs_T = transpose(input_cache);
    grad_weights = grad_output * inputs_T;

    grad_biases = Matrix(biases.rows, biases.cols);
    for(size_t i = 0; i < grad_output.rows; i++) {
        float sum = 0.0f;
        for(size_t j = 0; j < grad_output.cols; j++) {
            sum += grad_output[i][j];
        }
        grad_biases[i][0] = sum;
    }
    Matrix grad_inputs = transpose(weights) * grad_output; 

    return grad_inputs;
}

inline void Dense::update(float lr) {
    weights = weights - (grad_weights * lr);
    biases = biases - (grad_biases * lr);
}

// ReLU class
class ReLU : public Layer{
public:
    Matrix cache;
    Matrix forward(const Matrix &input) override; 
    Matrix backward(const Matrix &grad_output) override;
    void update(float lr) override {}
};

inline Matrix ReLU::forward(const Matrix &input) {
    cache = input;
    return relu(input);
}
inline Matrix ReLU::backward(const Matrix &grad_output) {
    Matrix grad(cache.rows, cache.cols);
    for(size_t i = 0; i < cache.rows; i++) {
        for(size_t j = 0; j < cache.cols; j++) {
            grad[i][j] = cache[i][j] > 0 ? grad_output[i][j] : 0.0f;
        }
    }
    return grad;
}

class MSELoss {
public:
    float forward(const Matrix &pred, const Matrix &target);
    Matrix backward(const Matrix &pred, const Matrix &target);
};

inline float MSELoss::forward(const Matrix &pred, const Matrix &target) {
    float sum = 0.f;
    int N = pred.rows * pred.cols;
    for(size_t i = 0; i < pred.rows; i++) {
       for(size_t j = 0; j < pred.rows; j++) {
            sum += std::pow((pred[i][j] - target[i][j]), 2);
       } 
    }
    return sum / N;
}

inline Matrix MSELoss::backward(const Matrix &pred, const Matrix &target) {
    if(pred.cols != target.cols || pred.rows != target.rows) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    Matrix delta(pred.rows, pred.cols);
    for(size_t i = 0; i < delta.rows; i++) {
        for(size_t j = 0; j < delta.cols; j++) {
            delta[i][j] = 2 * (pred[i][j] - target[i][j]);
        }
    }
    return delta;
}

#endif
