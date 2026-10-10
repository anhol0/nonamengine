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
    Tensor forward(const Tensor &m);
    void backward(const Tensor &grad);
    void update(float lr);

};

template<typename T, typename... Args>
void Model::add_layer(Args&& ...args) {
    layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
}
