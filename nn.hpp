#ifndef NN_HPP
#define NN_HPP

#include "matrix.hpp"
#include <iterator>

typedef struct BackpropResult {
    Matrix grad_weights;
    Matrix grad_biases;
} BackpropResult;

inline Matrix layer(
    const Matrix &inputs,
    const Matrix &weights,
    const Matrix &biases
) {
    Matrix m = weights * inputs;
    m = m + biases;
    return m;
}

inline float loss(const Matrix &prediction, const Matrix &target) {
    if(prediction.cols != target.cols || prediction.rows != target.rows) {
        throw std::invalid_argument("Invalid matrix dimensions");
    }
    float sum = 0.0f;
    for(int i = 0; i < prediction.rows; i++) {
        for(int j = 0; j < prediction.cols; j++) {
            float diff =  prediction[i][j] - target[i][j];
            sum += diff * diff;
        }
    }
    return sum / (prediction.rows * prediction.cols);
}

// loss(relu(layer(i, w, b)))
inline BackpropResult backprop(
        const Matrix &inputs,
        const Matrix &weights,
        const Matrix &target,
        const Matrix &prediction,
        const Matrix &biases
) {
    int N = prediction.rows * prediction.cols;

    Matrix delta(prediction.rows, prediction.cols);
    for(int i = 0; i < delta.rows; i++) {
        for(int j = 0; j < delta.cols; j++) {
            delta[i][j] = 2 * (prediction[i][j] - target[i][j]) / N;
        }
    }

    Matrix inputs_T = transpose(inputs);
    Matrix grad_weights = delta * inputs_T;

    Matrix grad_biases(biases.rows, biases.cols);
    for(int i = 0; i < delta.rows; i++) {
        float sum = 0.0f;
        for(int j = 0; j < delta.cols; j++) {
            sum += delta[i][j];
        }
        grad_biases[i][0] = sum;
    }
    return {grad_weights, grad_biases};
}

#endif
