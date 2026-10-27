#include "test_support.hpp"
#include "model.hpp"
#include <fstream>
#include <limits>

using Json = nlohmann::json;

void write_json(const std::filesystem::path& path, const Json& json) {
    std::ofstream out(path);
    out << json.dump(2);
    out.close();
    check(bool(out), "Could not write JSON fixture");
}

Json fixture(const std::filesystem::path& directory) {
    // Independent binary fixture: 2.0f, -1.0f, 0.5f in little-endian IEEE-754.
    constexpr unsigned char bytes[] = {0, 0, 0, 64, 0, 0, 128, 191, 0, 0, 0, 63};
    std::ofstream out(directory / "weights.bin", std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes), sizeof(bytes));
    out.close();
    check(bool(out), "Could not write binary fixture");
    return {
        {"format", "nonamengine"}, {"version", 1},
        {"weights", {{"file", "weights.bin"}, {"byte_order", "little"}, {"size", 12}}},
        {"layers", Json::array({{{"type", "dense"}, {"input_size", 2}, {"output_size", 1},
                                {"weights", "w"}, {"biases", "b"}}})},
        {"tensors", {
            {"w", {{"dtype", "float32"}, {"shape", {1, 2}}, {"offset", 0}, {"size", 8}}},
            {"b", {{"dtype", "float32"}, {"shape", {1, 1}}, {"offset", 8}, {"size", 4}}}
        }}
    };
}

void loading() {
    TestDirectory directory;
    auto json = fixture(directory.path);
    auto path = directory.path / "model.json";
    write_json(path, json);
    Model model;
    model.add_layer<ReLU>(); // A load replaces existing layers.
    check(model.load(path), "Independent fixture load");
    auto input = tensor({2, 2}, {1, -1, 3, 2});
    values(model.forward(input), {-0.5f, -3.5f});
    model.backward(tensor({1, 2}, {1, 1}));
    model.update(0.1f);
    values(model.forward(input), {-2.2f, -4.7f});
    json["weights"]["file"] = (directory.path / "weights.bin").string();
    json["layers"].push_back({{"type", "relu"}});
    json["layers"].push_back({{"type", "tanh"}});
    write_json(path, json);
    check(model.load(path), "Absolute path and activation load");
    values(model.forward(input), {0, 0});
    check(model.export_model(directory.path / "nested", "copy"), "Nested directory export");
    Model copy;
    check(copy.load(directory.path / "nested/copy.json"), "Round trip all layer types");
    equal(copy.forward(input), model.forward(input), 0);
}

void malformed() {
    TestDirectory directory;
    auto base = fixture(directory.path);
    auto path = directory.path / "model.json";
    Model model;
    model.add_layer<Dense>(tensor({1, 2}, {3, 4}), tensor({1, 1}, {5}));
    auto input = tensor({2, 1}, {1, 2});
    auto expected = model.forward(input);
    auto rejected = [&](const Json& bad) {
        write_json(path, bad);
        check(!model.load(path), "Invalid model accepted: " + bad.dump());
        equal(model.forward(input), expected, 0);
    };
    rejected(nullptr);
    for(const char* field : {"format", "version", "weights", "layers", "tensors"}) {
        auto bad = base;
        bad.erase(field);
        rejected(bad);
    }
    for(const Json& version : std::vector<Json>{-1, 0, 2, 1.5, "1", nullptr}) {
        auto bad = base; bad["version"] = version; rejected(bad);
    }
    for(const char* field : {"format", "layers", "tensors", "weights"}) {
        auto bad = base; bad[field] = 42; rejected(bad);
    }
    for(const auto& entry : std::vector<std::pair<std::string, Json>>{
            {"format", "other"}, {"layers", Json::object()}, {"tensors", Json::array()}}) {
        auto bad = base; bad[entry.first] = entry.second; rejected(bad);
    }
    for(const auto& entry : std::vector<std::pair<std::string, Json>>{
            {"file", "missing.bin"}, {"file", 12}, {"byte_order", "big"}, {"size", 11}, {"size", -1}}) {
        auto bad = base; bad["weights"][entry.first] = entry.second; rejected(bad);
    }
    for(const auto& entry : std::vector<std::pair<std::string, Json>>{
            {"type", "unknown"}, {"type", 1}, {"input_size", 0}, {"output_size", 0},
            {"input_size", -1}, {"input_size", 2.5}, {"weights", "absent"}, {"biases", nullptr}}) {
        auto bad = base; bad["layers"][0][entry.first] = entry.second; rejected(bad);
    }
    for(const char* field : {"type", "input_size", "output_size", "weights", "biases"}) {
        auto bad = base; bad["layers"][0].erase(field); rejected(bad);
    }
    for(const auto& entry : std::vector<std::pair<std::string, Json>>{
            {"dtype", "float16"}, {"dtype", nullptr}, {"shape", "1,2"},
            {"shape", {2, 1}}, {"shape", {-1, 2}}, {"shape", {1.0, 2}},
            {"offset", -1}, {"offset", 1.5}, {"offset", 13}, {"offset", 8},
            {"offset", std::numeric_limits<uint64_t>::max()}, {"size", 4}}) {
        auto bad = base; bad["tensors"]["w"][entry.first] = entry.second; rejected(bad);
    }
    for(const char* field : {"dtype", "shape", "offset", "size"}) {
        auto bad = base; bad["tensors"]["w"].erase(field); rejected(bad);
    }
    auto overflow = base;
    overflow["layers"][0]["input_size"] = std::numeric_limits<uint64_t>::max();
    overflow["tensors"]["w"]["shape"] = {1, std::numeric_limits<uint64_t>::max()};
    rejected(overflow);
    overflow["layers"][0]["output_size"] = 2;
    overflow["tensors"]["w"]["shape"] = {2, std::numeric_limits<uint64_t>::max()};
    rejected(overflow);
    check(!model.load(directory.path / "missing.json"), "Missing JSON must fail");
    { std::ofstream out(path); out << "{broken"; }
    check(!model.load(path), "Broken JSON must fail");
    { std::ofstream out(path); out << base.dump() << " trailing garbage"; }
    check(!model.load(path), "Trailing garbage must fail");
    write_json(path, base);
    std::filesystem::resize_file(directory.path / "weights.bin", 4);
    check(!model.load(path), "Truncated binary must fail");
    equal(model.forward(input), expected, 0);
}

void exporting() {
    TestDirectory directory;
    Model model;
    model.add_layer<Dense>(tensor({1, 2}, {2, -1}), tensor({1, 1}, {0.5f}));
    check(model.export_model(directory.path, "model"), "Export");
    std::ifstream bytes(directory.path / "model.bin", std::ios::binary);
    const std::vector<unsigned char> actual((std::istreambuf_iterator<char>(bytes)), {});
    check(actual == std::vector<unsigned char>{0, 0, 0, 64, 0, 0, 128, 191, 0, 0, 0, 63}, "Exact exported binary bytes");
    check(!model.export_model(directory.path, ""), "Empty export name");
    check(!model.export_model(directory.path, "../escape"), "Parent export name");
    check(!model.export_model(directory.path, "/absolute"), "Absolute export name");
    check(!model.export_model(directory.path / "model.bin", "x"), "File as directory");
    std::filesystem::create_directory(directory.path / "blocked.bin");
    check(!model.export_model(directory.path, "blocked"), "Blocked binary destination");
    std::filesystem::create_directory(directory.path / "blocked_json.json");
    check(!model.export_model(directory.path, "blocked_json"), "Blocked JSON destination");
    Model empty;
    check(empty.export_model(directory.path, "empty"), "Empty model export");
    check(empty.load(directory.path / "empty.json"), "Empty model load");
    auto input = tensor({1, 1}, {5});
    equal(empty.forward(input), input);
}

int main(int argc, char** argv) {
    return run_test([&] {
        check(argc == 2, "Expected case name");
        std::string name = argv[1];
        if(name == "loading") loading();
        else if(name == "malformed") malformed();
        else if(name == "exporting") exporting();
        else throw std::runtime_error("Unknown case");
    });
}
