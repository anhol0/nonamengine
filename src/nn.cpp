#include "include/nn.hpp"
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <vector>

Dense::Dense(size_t in_features, size_t out_features):
    weights({out_features, in_features}),
    biases({out_features, 1})
{

    float scale = std::sqrt(1 / static_cast<float>(in_features));

    std::random_device rand;
    std::mt19937 generator(rand());
    std::uniform_real_distribution<float> distribution(-scale, scale);

    weights.fill_with([&](){
        return distribution(generator);
    });

    biases.fill(0.0f);
}

Tensor Dense::forward(const Tensor& input) {
    input_cache = input;
    return weights.matmul(input) + biases;
}

Tensor Dense::backward(const Tensor &grad_output) {
    grad_weights = grad_output.matmul(input_cache->transposed());

    grad_biases.emplace(std::vector<uint64_t>({biases.shape()[0], 1}));

    for(size_t i = 0; i < grad_output.rows(); i++) {
        float sum = 0.0f;
        for(size_t j = 0; j < grad_output.cols(); j++) {
            sum += grad_output.at(i, j);
        }
        grad_biases->at(i, 0) = sum;
    }

    Tensor grad_inputs = weights.transposed().matmul(grad_output);

    return grad_inputs;
}

void Dense::update(float lr) {
    if (!grad_weights || !grad_biases) {
        throw std::logic_error(
            "Dense::update called before Dense::backward"
        );
    }

    weights = weights - (*grad_weights * lr);
    biases = biases - (*grad_biases * lr);
}


Tensor ReLU::forward(const Tensor &input) {
    cache = input;
    return tensor_operations::relu(input);
}

Tensor ReLU::backward(const Tensor &grad_output) {
    if(!cache)
            throw std::logic_error("ReLU::Cache was not initialized");

    Tensor grad({cache->rows(), cache->cols()});
    for(size_t i = 0; i < cache->rows(); i++) {
        for(size_t j = 0; j < cache->cols(); j++) {
            grad.at(i, j) = cache->at(i, j) > 0 ? grad_output.at(i, j) : 0.0f;
        }
    }
    return grad;
}


Tensor Tanh::forward(const Tensor &input) {
    cache = input;
    if(!cache)
        throw std::logic_error("Tanh::Cache was not initialized");

    Tensor out({cache->rows(), cache->cols()});
    for(size_t i = 0; i < cache->rows(); i++) {
        for(size_t j = 0; j < cache->cols(); j++) {
            out.at(i, j) = std::tanh(cache->at(i, j));
        }
    }
    return out;
}

Tensor Tanh::backward(const Tensor &grad_output) {
    if(!cache)
        throw std::logic_error("Tanh::Cache was not initialized");

    Tensor grad({cache->rows(), cache->cols()});
    for(size_t i = 0; i < grad.rows(); i++) {
        for(size_t j = 0; j < cache->cols(); j++) {
            float t = std::tanh(cache->at(i, j));
            grad.at(i, j) = grad_output.at(i, j) * (1 - t * t);
        }
    }
    return grad;
}


float MSELoss::forward(const Tensor &pred, const Tensor &target) {
    if(pred.cols() != target.cols() || pred.rows() != target.rows()) {
        throw std::invalid_argument("Invalid Tensor dimensions");
    }
    float sum = 0.f;
    int N = pred.rows() * pred.cols();
    for(size_t i = 0; i < pred.rows(); i++) {
       for(size_t j = 0; j < pred.cols(); j++) {
           float diff = pred.at(i, j) - target.at(i, j);
            sum += diff * diff;
       }
    }
    return sum / N;
}

Tensor MSELoss::backward(const Tensor &pred, const Tensor &target) {
    if(pred.cols() != target.cols() || pred.rows() != target.rows()) {
        throw std::invalid_argument("Invalid Tensor dimensions");
    }
    Tensor delta({pred.rows(), pred.cols()});
    size_t N = pred.rows() * pred.cols();
    for(size_t i = 0; i < delta.rows(); i++) {
        for(size_t j = 0; j < delta.cols(); j++) {
            delta.at(i, j) = 2 * (pred.at(i, j) - target.at(i, j)) / N;
        }
    }
    return delta;
}
