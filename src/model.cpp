#include "include/model.hpp"
#include <array>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
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

const nlohmann::json& required_member(
    const nlohmann::json& object,
    std::string_view name
) {
    if(!object.is_object()) {
        throw std::invalid_argument("Expected a JSON object");
    }

    const auto member = object.find(std::string(name));
    if(member == object.end()) {
        throw std::invalid_argument(
            "Missing required field: " + std::string(name)
        );
    }
    return *member;
}

std::string required_string(
    const nlohmann::json& object,
    std::string_view name
) {
    const nlohmann::json& value = required_member(object, name);
    if(!value.is_string()) {
        throw std::invalid_argument(
            "Field must be a string: " + std::string(name)
        );
    }
    return value.get<std::string>();
}

std::uint64_t required_unsigned(
    const nlohmann::json& object,
    std::string_view name
) {
    const nlohmann::json& value = required_member(object, name);
    if(value.is_number_unsigned()) {
        return value.get<std::uint64_t>();
    }
    if(value.is_number_integer()) {
        const std::int64_t signed_value = value.get<std::int64_t>();
        if(signed_value >= 0) {
            return static_cast<std::uint64_t>(signed_value);
        }
    }
    throw std::invalid_argument(
        "Field must be a non-negative integer: " + std::string(name)
    );
}

nlohmann::json parse_description(
    const std::filesystem::path& description_file
) {
    std::ifstream input(description_file);
    if(!input) {
        throw std::runtime_error(
            "Could not open model description: " + description_file.string()
        );
    }

    nlohmann::json description;
    input >> description;
    return description;
}

std::vector<std::uint64_t> parse_shape(const nlohmann::json& value) {
    if(!value.is_array()) {
        throw std::invalid_argument("Tensor shape must be an array");
    }

    std::vector<std::uint64_t> shape;
    shape.reserve(value.size());
    for(const nlohmann::json& dimension : value) {
        std::uint64_t extent = 0;
        if(dimension.is_number_unsigned()) {
            extent = dimension.get<std::uint64_t>();
        } else if(dimension.is_number_integer()) {
            const std::int64_t signed_extent = dimension.get<std::int64_t>();
            if(signed_extent < 0) {
                throw std::invalid_argument(
                    "Tensor dimensions cannot be negative"
                );
            }
            extent = static_cast<std::uint64_t>(signed_extent);
        } else {
            throw std::invalid_argument(
                "Tensor dimensions must be integers"
            );
        }
        shape.push_back(extent);
    }
    return shape;
}

float read_float32_little_endian(std::istream& input) {
    std::array<unsigned char, 4> bytes {};
    input.read(
        reinterpret_cast<char*>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())
    );
    if(!input) {
        throw std::runtime_error("Could not read tensor data");
    }

    const std::uint32_t bits =
        static_cast<std::uint32_t>(bytes[0]) |
        (static_cast<std::uint32_t>(bytes[1]) << 8U) |
        (static_cast<std::uint32_t>(bytes[2]) << 16U) |
        (static_cast<std::uint32_t>(bytes[3]) << 24U);
    return std::bit_cast<float>(bits);
}

Tensor load_tensor(
    const nlohmann::json& tensors,
    const std::string& tensor_name,
    const std::vector<std::uint64_t>& expected_shape,
    std::ifstream& weights_input,
    std::uint64_t weights_size
) {
    const nlohmann::json& tensor = required_member(tensors, tensor_name);
    if(required_string(tensor, "dtype") != "float32") {
        throw std::invalid_argument(
            "Unsupported dtype for tensor: " + tensor_name
        );
    }

    const std::vector<std::uint64_t> shape =
        parse_shape(required_member(tensor, "shape"));
    if(shape != expected_shape) {
        throw std::invalid_argument(
            "Unexpected shape for tensor: " + tensor_name
        );
    }

    std::uint64_t element_count = 1;
    for(const std::uint64_t extent : shape) {
        if(extent != 0 &&
           element_count >
               std::numeric_limits<std::uint64_t>::max() / extent) {
            throw std::overflow_error(
                "Element count overflow for tensor: " + tensor_name
            );
        }
        element_count *= extent;
    }
    if(element_count >
       std::numeric_limits<std::uint64_t>::max() / sizeof(float)) {
        throw std::overflow_error(
            "Byte size overflow for tensor: " + tensor_name
        );
    }

    const std::uint64_t expected_size = element_count * sizeof(float);
    const std::uint64_t offset = required_unsigned(tensor, "offset");
    const std::uint64_t size = required_unsigned(tensor, "size");
    if(size != expected_size) {
        throw std::invalid_argument(
            "Incorrect byte size for tensor: " + tensor_name
        );
    }
    if(offset > weights_size || size > weights_size - offset) {
        throw std::invalid_argument(
            "Tensor is outside the weights file: " + tensor_name
        );
    }
    if(offset > static_cast<std::uint64_t>(
            std::numeric_limits<std::streamoff>::max())) {
        throw std::overflow_error(
            "Tensor offset is too large: " + tensor_name
        );
    }

    weights_input.clear();
    weights_input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if(!weights_input) {
        throw std::runtime_error(
            "Could not seek to tensor: " + tensor_name
        );
    }

    Tensor result(shape);
    for(float& value : result.data()) {
        value = read_float32_little_endian(weights_input);
    }
    return result;
}

std::vector<std::unique_ptr<Layer>> load_layers(
    const nlohmann::json& description,
    std::ifstream& weights_input,
    std::uint64_t weights_size
) {
    const nlohmann::json& layer_descriptions =
        required_member(description, "layers");
    const nlohmann::json& tensors = required_member(description, "tensors");
    if(!layer_descriptions.is_array()) {
        throw std::invalid_argument("Layers must be an array");
    }
    if(!tensors.is_object()) {
        throw std::invalid_argument("Tensors must be an object");
    }

    std::vector<std::unique_ptr<Layer>> loaded_layers;
    loaded_layers.reserve(layer_descriptions.size());

    for(const nlohmann::json& layer_description : layer_descriptions) {
        const std::string type = required_string(layer_description, "type");
        if(type == "relu") {
            loaded_layers.push_back(std::make_unique<ReLU>());
            continue;
        }
        if(type == "tanh") {
            loaded_layers.push_back(std::make_unique<Tanh>());
            continue;
        }
        if(type != "dense") {
            throw std::invalid_argument("Unsupported layer type: " + type);
        }

        const std::uint64_t input_size =
            required_unsigned(layer_description, "input_size");
        const std::uint64_t output_size =
            required_unsigned(layer_description, "output_size");
        if(input_size == 0 || output_size == 0 ||
           input_size > std::numeric_limits<std::size_t>::max() ||
           output_size > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("Invalid Dense layer dimensions");
        }

        const std::string weights_name =
            required_string(layer_description, "weights");
        const std::string biases_name =
            required_string(layer_description, "biases");

        Tensor loaded_weights = load_tensor(
            tensors,
            weights_name,
            {output_size, input_size},
            weights_input,
            weights_size
        );
        Tensor loaded_biases = load_tensor(
            tensors,
            biases_name,
            {output_size, 1},
            weights_input,
            weights_size
        );

        loaded_layers.push_back(std::make_unique<Dense>(
            std::move(loaded_weights),
            std::move(loaded_biases)
        ));
    }

    return loaded_layers;
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

bool Model::load(const std::filesystem::path& description_file) {
    namespace fs = std::filesystem;

    try {
        const nlohmann::json description =
            parse_description(description_file);
        if(required_string(description, "format") != "nonamengine") {
            throw std::invalid_argument("Unsupported model format");
        }
        if(required_unsigned(description, "version") != 1) {
            throw std::invalid_argument("Unsupported model version");
        }

        const nlohmann::json& weights =
            required_member(description, "weights");
        if(required_string(weights, "byte_order") != "little") {
            throw std::invalid_argument("Unsupported weights byte order");
        }

        fs::path weights_file(required_string(weights, "file"));
        if(weights_file.is_relative()) {
            weights_file = description_file.parent_path() / weights_file;
        }

        std::ifstream weights_input(
            weights_file,
            std::ios::binary | std::ios::ate
        );
        if(!weights_input) {
            throw std::runtime_error(
                "Could not open weights file: " + weights_file.string()
            );
        }

        const std::streampos end_position = weights_input.tellg();
        if(end_position < 0) {
            throw std::runtime_error(
                "Could not determine weights file size"
            );
        }
        const std::uint64_t weights_size =
            static_cast<std::uint64_t>(end_position);
        if(required_unsigned(weights, "size") != weights_size) {
            throw std::invalid_argument(
                "Weights file size does not match its description"
            );
        }

        std::vector<std::unique_ptr<Layer>> loaded_layers =
            load_layers(description, weights_input, weights_size);
        layers = std::move(loaded_layers);
        return true;
    } catch(const std::exception& e) {
        std::cerr << "Could not load model: " << e.what() << '\n';
        return false;
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
