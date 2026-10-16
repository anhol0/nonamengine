#include "model.hpp"
#include "nn.hpp"

#include <nlohmann/json.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace {

bool require(bool condition, const char* message) {
    if(!condition) {
        std::cerr << message << '\n';
    }
    return condition;
}

} // namespace

int main() {
    namespace fs = std::filesystem;

    const auto unique_suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const fs::path export_directory = fs::temp_directory_path() /
        ("nonamengine-export-test-" + std::to_string(unique_suffix));

    Model model;
    model.add_layer<Dense>(2, 32);
    model.add_layer<ReLU>();
    model.add_layer<Dense>(32, 1);

    if(!require(model.export_model(export_directory, "multiply"),
                "Model export failed")) {
        return 1;
    }

    std::ifstream description_input(export_directory / "multiply.json");
    if(!require(static_cast<bool>(description_input),
                "Could not open exported JSON")) {
        return 1;
    }

    const nlohmann::json description =
        nlohmann::json::parse(description_input);
    bool valid = true;
    valid &= require(description.at("format") == "nonamengine",
                     "Unexpected model format");
    valid &= require(description.at("version") == 1,
                     "Unexpected model version");
    valid &= require(description.at("layers").size() == 3,
                     "Unexpected layer count");
    valid &= require(description.at("tensors").size() == 4,
                     "Unexpected tensor count");
    valid &= require(description.at("weights").at("file") == "multiply.bin",
                     "Unexpected weights file name");
    valid &= require(description.at("weights").at("size") == 516,
                     "Unexpected weights byte count");

    const auto& first_layer = description.at("layers").at(0);
    valid &= require(first_layer.at("input_size") == 2,
                     "Dense input size was exported incorrectly");
    valid &= require(first_layer.at("output_size") == 32,
                     "Dense output size was exported incorrectly");
    valid &= require(first_layer.at("weights") == "layers.0.weights",
                     "Dense weights reference is incorrect");
    valid &= require(first_layer.at("biases") == "layers.0.biases",
                     "Dense biases reference is incorrect");

    const auto& first_weights =
        description.at("tensors").at("layers.0.weights");
    valid &= require(first_weights.at("shape") ==
                         nlohmann::json::array({32, 2}),
                     "Dense weights shape is incorrect");
    valid &= require(first_weights.at("offset") == 0,
                     "Dense weights offset is incorrect");
    valid &= require(first_weights.at("size") == 256,
                     "Dense weights size is incorrect");

    const auto& last_biases =
        description.at("tensors").at("layers.2.biases");
    valid &= require(last_biases.at("shape") ==
                         nlohmann::json::array({1, 1}),
                     "Output biases shape is incorrect");
    valid &= require(last_biases.at("offset") == 512,
                     "Output biases offset is incorrect");
    valid &= require(last_biases.at("size") == 4,
                     "Output biases size is incorrect");

    valid &= require(fs::file_size(export_directory / "multiply.bin") == 516,
                     "Binary weights file has the wrong size");

    Tensor input({2, 3});
    input.at(0, 0) = 0.25f;
    input.at(1, 0) = 0.11f;
    input.at(0, 1) = -0.5f;
    input.at(1, 1) = 0.75f;
    input.at(0, 2) = 1.0f;
    input.at(1, 2) = -1.0f;
    const Tensor expected = model.forward(input);

    Model loaded_model;
    valid &= require(
        loaded_model.load(export_directory / "multiply.json"),
        "Could not load exported model"
    );
    if(valid) {
        const Tensor actual = loaded_model.forward(input);
        valid &= require(actual.shape().size() == expected.shape().size(),
                         "Loaded model output rank differs");
        valid &= require(actual.data().size() == expected.data().size(),
                         "Loaded model output size differs");
        if(actual.data().size() == expected.data().size()) {
            for(std::size_t i = 0; i < actual.data().size(); ++i) {
                valid &= require(actual.data()[i] == expected.data()[i],
                                 "Loaded model output differs");
            }
        }
    }

    fs::remove_all(export_directory);
    return valid ? 0 : 1;
}
