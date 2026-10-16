#pragma once

#include "nn.hpp"
#include "tensor.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string_view>
#include <utility>
#include <vector>

class Model {
private:
    std::vector<std::unique_ptr<Layer>> layers;
    void save_model(
        const std::filesystem::path& description_file,
        const std::filesystem::path& model_file,
        std::uint32_t version
    ) const;

public:
    template<typename T, typename... Args>
    void add_layer(Args&& ...args);
    Tensor forward(const Tensor &m);
    void backward(const Tensor &grad);
    void update(float lr);
    bool load(const std::filesystem::path& description_file);
    bool export_model(
        const std::filesystem::path& dir_path,
        std::string_view name
    ) const;
};

template<typename T, typename... Args>
void Model::add_layer(Args&& ...args) {
    layers.push_back(std::make_unique<T>(std::forward<Args>(args)...));
}
