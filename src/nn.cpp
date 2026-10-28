#include "include/nn.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

Dense::Dense(size_t in_features, size_t out_features):
    weights({out_features, in_features}),
    biases({out_features, 1})
{
    if(in_features == 0 || out_features == 0) {
        throw std::invalid_argument("Dense dimensions must be positive");
    }
    float scale = std::sqrt(1 / static_cast<float>(in_features));

    std::random_device rand;
    std::mt19937 generator(rand());
    std::uniform_real_distribution<float> distribution(-scale, scale);

    weights.fill_with([&](){
        return distribution(generator);
    });

    biases.fill(0.0f);
}

Dense::Dense(Tensor initial_weights, Tensor initial_biases) :
    weights(std::move(initial_weights)),
    biases(std::move(initial_biases))
{
    if(weights.rank() != 2 || biases.rank() != 2 ||
       biases.rows() != weights.rows() || biases.cols() != 1 ||
       weights.rows() == 0 || weights.cols() == 0) {
        throw std::invalid_argument("Invalid Dense parameter shapes");
    }
}

Tensor Dense::forward(const Tensor& input) {
    input_cache = input;
    return weights.matmul(input) + biases;
}

Tensor Dense::backward(const Tensor &grad_output) {
    if(!input_cache) {
        throw std::logic_error("Dense::backward called before Dense::forward");
    }
    if(grad_output.rank() != 2 ||
       grad_output.rows() != weights.rows() ||
       grad_output.cols() != input_cache->cols()) {
        throw std::invalid_argument("Invalid Dense output gradient shape");
    }
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

nlohmann::json Dense::export_json() const {
    return {
        {"type", "dense"},
        {"input_size", weights.cols()},
        {"output_size", weights.rows()},
        {"weights", nullptr},
        {"biases", nullptr},
    };
}

std::vector<NamedTensor> Dense::parameters() const {
    return {
        {"weights", std::cref(weights)},
        {"biases", std::cref(biases)},
    };
}


Tensor ReLU::forward(const Tensor &input) {
    cache = input;
    return tensor_operations::relu(input);
}

Tensor ReLU::backward(const Tensor &grad_output) {
    if(!cache)
            throw std::logic_error("ReLU::Cache was not initialized");

    if(!std::ranges::equal(grad_output.shape(), cache->shape())) {
        throw std::invalid_argument("Invalid ReLU output gradient shape");
    }

    Tensor grad({cache->rows(), cache->cols()});
    for(size_t i = 0; i < cache->rows(); i++) {
        for(size_t j = 0; j < cache->cols(); j++) {
            grad.at(i, j) = cache->at(i, j) > 0 ? grad_output.at(i, j) : 0.0f;
        }
    }
    return grad;
}

nlohmann::json ReLU::export_json() const {
    return {
        {"type", "relu"},
    };
}

Tensor Tanh::forward(const Tensor &input) {
    cache = input;

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

    if(!std::ranges::equal(grad_output.shape(), cache->shape())) {
        throw std::invalid_argument("Invalid Tanh output gradient shape");
    }

    Tensor grad({cache->rows(), cache->cols()});
    for(size_t i = 0; i < grad.rows(); i++) {
        for(size_t j = 0; j < cache->cols(); j++) {
            float t = std::tanh(cache->at(i, j));
            grad.at(i, j) = grad_output.at(i, j) * (1 - t * t);
        }
    }
    return grad;
}

nlohmann::json Tanh::export_json() const {
    return {
        {"type", "tanh"},
    };
}


float MSELoss::forward(const Tensor &pred, const Tensor &target) {
    if(pred.cols() != target.cols() || pred.rows() != target.rows()) {
        throw std::invalid_argument("Invalid Tensor dimensions");
    }
    if(pred.data().empty()) {
        throw std::invalid_argument("MSE requires non-empty tensors");
    }
    float sum = 0.f;
    const std::size_t N = pred.data().size();
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
    if(pred.data().empty()) {
        throw std::invalid_argument("MSE requires non-empty tensors");
    }
    Tensor delta({pred.rows(), pred.cols()});
    const std::size_t N = pred.data().size();
    for(size_t i = 0; i < delta.rows(); i++) {
        for(size_t j = 0; j < delta.cols(); j++) {
            delta.at(i, j) = 2 * (pred.at(i, j) - target.at(i, j)) / N;
        }
    }
    return delta;
}
