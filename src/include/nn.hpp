#pragma once

#include "matrix.hpp"
#include <optional>

class Layer {
public:
    virtual Tensor forward(const Tensor &input) = 0;
    virtual Tensor backward(const Tensor &grad_output) = 0;
    virtual void update(float lr) = 0;
    virtual ~Layer() = default;
};

// Base Dense layer
class Dense : public Layer {
public:
    Tensor weights, biases;
    std::optional<Tensor> grad_weights, grad_biases;
    std::optional<Tensor> input_cache;

    Dense(size_t in_features, size_t out_features);
    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;
    void update(float lr) override;
};

// ReLU class
class ReLU : public Layer {
public:
    std::optional<Tensor> cache;
    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;
    void update(float lr) override {(void)lr;}
};

// Tanh class
class Tanh : public Layer {
public:
    std::optional<Tensor> cache;
    Tensor forward(const Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;
    void update(float lr) override {(void)lr;}
};

// MSE loss class for error calculation
class MSELoss {
public:
    float forward(const Tensor &pred, const Tensor &target);
    Tensor backward(const Tensor &pred, const Tensor &target);
};
