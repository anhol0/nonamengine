#include "include/model.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <nlohmann/json.hpp>

namespace {

void write_float32_little_endian(std::ostream& output, float value) {
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    static_assert(std::numeric_limits<float>::is_iec559);

    const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
    const std::array<char, 4> bytes {
        static_cast<char>(bits & 0xffU),
        static_cast<char>((bits >> 8U) & 0xffU),
        static_cast<char>((bits >> 16U) & 0xffU),
        static_cast<char>((bits >> 24U) & 0xffU),
    };
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

nlohmann::json tensor_shape(const Tensor& tensor) {
    nlohmann::json shape = nlohmann::json::array();
    for(const std::size_t extent : tensor.shape()) {
        shape.push_back(extent);
    }
    return shape;
}

} // namespace

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

void Model::save_model(
    const std::filesystem::path& description_file,
    const std::filesystem::path& model_file,
    std::uint32_t version
) const {
    namespace nl = nlohmann;

    std::ofstream weights_output(model_file, std::ios::binary);
    if(!weights_output) {
        throw std::runtime_error(
            "Could not open weights file: " + model_file.string()
        );
    }

    nl::json json;
    json["format"] = "nonamengine";
    json["version"] = version;
    json["layers"] = nl::json::array();
    json["tensors"] = nl::json::object();

    std::uint64_t offset = 0;
    for(std::size_t i = 0; i < layers.size(); ++i) {
        nl::json layer_json = layers[i]->export_json();

        for(const auto& [parameter_name, tensor_reference]
            : layers[i]->parameters()) {
            const Tensor& tensor = tensor_reference.get();
            const std::uint64_t element_count = tensor.data().size();
            if(element_count >
               std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
                throw std::overflow_error("Tensor byte size is too large");
            }

            const std::uint64_t byte_size = element_count * sizeof(float);
            if(offset > std::numeric_limits<std::uint64_t>::max() - byte_size) {
                throw std::overflow_error("Weights file is too large");
            }

            const std::string tensor_name =
                "layers." + std::to_string(i) + "." +
                std::string(parameter_name);

            layer_json[std::string(parameter_name)] = tensor_name;
            json["tensors"][tensor_name] = {
                {"dtype", "float32"},
                {"shape", tensor_shape(tensor)},
                {"offset", offset},
                {"size", byte_size},
            };

            for(const float value : tensor.data()) {
                write_float32_little_endian(weights_output, value);
            }
            offset += byte_size;
        }

        json["layers"].push_back(std::move(layer_json));
    }

    weights_output.close();
    if(!weights_output) {
        throw std::runtime_error(
            "Could not write weights file: " + model_file.string()
        );
    }

    json["weights"] = {
        {"file", model_file.filename().string()},
        {"byte_order", "little"},
        {"size", offset},
    };

    std::ofstream description_output(description_file);
    if(!description_output) {
        throw std::runtime_error(
            "Could not open model description: " + description_file.string()
        );
    }
    description_output << json.dump(2) << '\n';
    description_output.close();
    if(!description_output) {
        throw std::runtime_error(
            "Could not write model description: " + description_file.string()
        );
    }
}

bool Model::export_model(
    const std::filesystem::path& dir_path,
    std::string_view name
) const {
    namespace fs = std::filesystem;

    const fs::path model_name {std::string(name)};
    if(name.empty() || model_name.has_parent_path() || model_name.has_root_path()) {
        std::cerr << "Model name must be a non-empty file name\n";
        return false;
    }

    try {
        if(fs::exists(dir_path) && !fs::is_directory(dir_path)) {
            std::cerr << dir_path << " is not a directory\n";
            return false;
        }
        fs::create_directories(dir_path);

        const fs::path description_file =
            dir_path / (model_name.string() + ".json");
        const fs::path weights_file =
            dir_path / (model_name.string() + ".bin");

        save_model(description_file, weights_file, 1);
        return true;
    } catch(const std::exception& e) {
        std::cerr << "Could not export model: " << e.what() << '\n';
        return false;
    }
}
