#include "include/model.hpp"

Tensor Model::forward(const Tensor &m) {
    Tensor pred = m;
    for(const auto &e: layers) {
        pred = e->forward(pred);
    }
    return pred;
}

void Model::backward(const Tensor &grad) {
    Tensor grad_act = grad;
    for(size_t i = layers.size(); i-- > 0;) {
         grad_act = layers[i]->backward(grad_act);
    }
}

void Model::update(float lr) {
    for(auto &e: layers) {
        e->update(lr);
    }
}
