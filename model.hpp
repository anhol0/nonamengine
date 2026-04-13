#ifndef MODEL_HPP
#define MODEL_HPP

#include "nn.hpp"
#include "matrix.hpp"
#include <memory>
#include <utility>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers; 
public:
    template<typename T, typename... Args>
    void add_layer(Args&& ...args);
    Matrix forward(const Matrix &m);
    Matrix gradient(const Matrix &pred, const Matrix& target);
    void backward(const Matrix &grad);
    void update(float lr);
};

template<typename T, typename... Args>
inline void Model::add_layer(Args&& ...args) {
    layers.push_back(std::make_unique<T>(std::forward<Args>(args)...)); 
}

inline Matrix Model::forward(const Matrix &m) {
    Matrix pred = m;
    for(const auto &e: layers) {
        pred = e->forward(pred);
    }
    return pred;
}

inline Matrix Model::gradient(const Matrix &pred, const Matrix &target) {
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

inline void Model::backward(const Matrix &grad) {
    Matrix grad_act = grad;
    for(size_t i = layers.size(); i-- > 0;) {
         grad_act = layers[i]->backward(grad_act);
    }
}

inline void Model::update(float lr) {
    for(auto &e: layers) {
        e->update(lr);
    } 
}

#endif
