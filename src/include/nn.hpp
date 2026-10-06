#pragma once

#include "matrix.hpp"

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

// ReLU class
class ReLU : public Layer{
public:
    Matrix cache;
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &grad_output) override;
    void update(float lr) override {(void)lr;}
};

// Tanh class
class Tanh : public Layer {
public:
    Matrix cache;
    Matrix forward(const Matrix &input) override;
    Matrix backward(const Matrix &grad_output) override;
    void update(float lr) override {(void)lr;}
};

// MSE loss class for error calculation
class MSELoss {
public:
    float forward(const Matrix &pred, const Matrix &target);
    Matrix backward(const Matrix &pred, const Matrix &target);
};
