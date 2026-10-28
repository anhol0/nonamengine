#pragma once

// Internal persistence functions, not part of the public engine interface.
#include "include/nn.hpp"

#include <cstdint>
#include <iosfwd>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace model_io {

nlohmann::json read_description(std::istream& input);
std::vector<std::unique_ptr<Layer>> read_layers(
    const nlohmann::json& description, std::istream& weights);
std::uint64_t tensor_end_offset(std::uint64_t elements, std::uint64_t offset);
nlohmann::json write_weights(
    std::span<const std::unique_ptr<Layer>> layers, std::ostream& output,
    std::string_view filename, std::uint32_t version);
void write_description(std::ostream& output, const nlohmann::json& description);
void close_output(std::ofstream& output);

} // namespace model_io
