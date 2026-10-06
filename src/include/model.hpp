#pragma once

#include "nn.hpp"
#include "matrix.hpp"

#include <memory>
#include <utility>
#include <vector>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
public:
    template<typename T, typename... Args>
    void add_layer(Args&& ...args);
    Matrix forward(const Matrix &m);
    void backward(const Matrix &grad);
    void update(float lr);
};

template<typename T, typename... Args>
void Model::add_layer(Args&& ...args) {
    layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
}
